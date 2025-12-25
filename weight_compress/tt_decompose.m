% Decompose ALL MobileNetV2 layers
% Requires: 
% 1. TT-Toolbox (https://github.com/oseledets/TT-Toolbox) 
% 2. Sandia Tensor Toolbox (https://www.tensortoolbox.org/)
% Authors: Yu-Sheng Tzou
% Date: 2025.12.13

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('MobileNetV2 Tensor Decomposition: Tucker (Conv) + TT (FC)\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

% 1. Find weights
weight_files = dir('weights/mobilenet_*.bin');
if isempty(weight_files)
    error('No weight files found! Run extract_mobilenet.py first.');
end

% 2. Settings
tucker_tol = 0.1;   % ⚠️ Conv 容忍度 (10% 誤差)
tt_tol = 0.1;       % ⚠️ FC TT 容忍度 (10% 誤差)
svd_rank_ratio = 0.1; % ⚠️ FC SVD 壓縮秩比例 (使用原始秩的 10%)
results = struct('layer', {}, 'method', {}, 'params', {}, 'compression', {}, 'error', {});

% 3. Process each layer
for file_idx = 1:length(weight_files)
    filename = weight_files(file_idx).name;
    base_name = filename(1:end-4);
    
    % Skip if it's a core file (previously generated)
    if contains(base_name, '_core') || contains(base_name, '_tucker') || contains(base_name, '_tt') || contains(base_name, '_svd')
        continue;
    end
    
    % Load Info
    info_file = ['weights/' base_name '_info.txt'];
    if ~exist(info_file, 'file'), continue; end
    
    fid = fopen(info_file, 'r');
    info_lines = textscan(fid, '%s', 'Delimiter', '\n'); 
    fclose(fid);
    
    % --- Layer Type Detection (Standard if-else) ---
    raw_type_line = info_lines{1}{1};
    if contains(raw_type_line, 'Conv')
        layer_type = 'Conv';
    elseif contains(raw_type_line, 'FC')
        layer_type = 'FC';
    else
        layer_type = 'Unknown';
    end
    % -------------------------------------------------------
    
    % Parse Shape
    shape_str = regexp(info_lines{1}{2}, '\[(.*?)\]', 'tokens');
    shape = str2num(shape_str{1}{1});
    
    % Load Weights
    fid = fopen(['weights/' filename], 'rb');
    uncompressed_weights_flat = fread(fid, inf, 'float32'); 
    fclose(fid);
    
    % Reshape (MATLAB uses column-major)
    uncompressed_weights_tensor = reshape(uncompressed_weights_flat, shape);
    original_size = numel(uncompressed_weights_tensor);
    
    fprintf('───────────────────────────────────────────────────────────\n');
    fprintf('[%d/%d] %s (%s)\n', file_idx, length(weight_files), base_name, layer_type);
    
    try
        if strcmp(layer_type, 'Conv')
            %% Tucker Decomposition
            fprintf('  Method: Tucker (Sandia Toolbox)\n');
            
            % 步驟 A: 將 double 轉為 tensor 物件
            T_data = tensor(uncompressed_weights_tensor);
            
            % 步驟 B: 自動計算 Ranks
            est_ranks = zeros(1, ndims(T_data));
            for d = 1:ndims(T_data)
                Mat_d = double(tenmat(T_data, d)); 
                s = svd(Mat_d, 'econ');
                energy = cumsum(s.^2) / sum(s.^2);
                r = find(energy >= (1 - tucker_tol), 1);
                if isempty(r), r = length(s); end
                est_ranks(d) = r;
            end
            
            fprintf('  Auto-Ranks: [%s] (Tol: %.0e)\n', num2str(est_ranks), tucker_tol);
            
            % 步驟 C: 執行 Tucker ALS
            T_tucker = tucker_als(T_data, est_ranks);
            
            % 步驟 D: 重構與統計
            reconstructed = double(T_tucker); 
            
            % 計算儲存量 (Core + Factors)
            core_size = numel(T_tucker.core);
            factor_size = 0;
            for k=1:length(T_tucker.U)
                factor_size = factor_size + numel(T_tucker.U{k}); 
            end
            compressed_size = core_size + factor_size;
            
            % Save and Print
            save(sprintf('weights/%s_tucker.mat', base_name), 'T_tucker', 'est_ranks');
            
            compression = original_size / compressed_size;
            rel_error = norm(uncompressed_weights_tensor(:) - reconstructed(:)) / norm(uncompressed_weights_tensor(:));
            
            fprintf('  Original: %d | Compressed: %d\n', original_size, compressed_size);
            fprintf('  Ratio: %.2fx | Error: %.2e', compression, rel_error);
            
            if compression < 1.0
                fprintf(' ❌ EXPANSION\n');
            elseif compression > 3.0
                 fprintf(' ✅ GOOD\n');
            else
                 fprintf(' ⚠ MARGINAL\n');
            end
            
            results(end+1) = struct('layer', base_name, 'method', 'Tucker', ...
                'params', original_size, 'compression', compression, 'error', rel_error);
            
        elseif strcmp(layer_type, 'FC')
            
            % ----------------------------------------------------
            % TT Decomposition (TT-Toolbox)
            % ----------------------------------------------------
            fprintf('  Method: TT Decomposition\n');
            
            w_target = uncompressed_weights_tensor;
            
            % Execute TT Decomposition
            tt_res = tt_tensor(w_target, tt_tol);
            disp(tt_res);
            fprintf('TT-Toolbox: size(tt_res) = %d\n', size(tt_res));

            % Reconstruction for verification
            reconstructed_tt = full(tt_res); % Shape: [y1, x1, y2, x2, ...]
            
            compressed_size_tt = numel(tt_res.core); % TT core total params
            
            % TT Result Calculation and Print
            rel_error_tt = norm(uncompressed_weights_tensor(:) - reconstructed_tt(:)) / norm(uncompressed_weights_tensor(:));
            compression_tt = original_size / compressed_size_tt;
            
            fprintf('  [TT] Original: %d | Compressed: %d\n', original_size, compressed_size_tt);
            fprintf('  [TT] Ratio: %.2fx | Error: %.2e', compression_tt, rel_error_tt);
            
            if compression_tt < 1.0
                fprintf(' ❌ EXPANSION\n');
            elseif compression_tt > 3.0
                 fprintf(' ✅ GOOD\n');
            else
                 fprintf(' ⚠ MARGINAL\n');
            end
            
            % Save TT
            save(sprintf('weights/%s_tt.mat', base_name), 'tt_res');
            results(end+1) = struct('layer', base_name, 'method', 'TT', ...
                'params', original_size, 'compression', compression_tt, 'error', rel_error_tt);
            
        else
            fprintf('  Skipping unknown layer type.\n');
            continue;
        end
            
    catch err
        fprintf('  ❌ FAILED: %s\n', err.message);
        fprintf('  Line: %d\n', err.stack(1).line);
    end
    fprintf('\n');
end

