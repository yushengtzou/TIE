% MATLAB Script: Test TT-decomposition on MobileNet Conv weights
% Compare with C++ implementation

fprintf('=== MobileNet Conv Weight TT-Decomposition ===\n\n');

% Add TT-Toolbox
addpath('../TT-Toolbox');

%% List available weight files
weight_files = dir('test_data/mobilenet_*.bin');

if isempty(weight_files)
    fprintf('No MobileNet weights found!\n');
    fprintf('Run: python3 python_extract_mobilenet.py\n');
    return;
end

fprintf('Found %d MobileNet weight files:\n', length(weight_files));
for i = 1:length(weight_files)
    fprintf('  %d. %s\n', i, weight_files(i).name);
end
fprintf('\n');

%% Test each layer
for file_idx = 1:length(weight_files)
    filename = weight_files(file_idx).name;
    base_name = filename(1:end-4);  % Remove .bin
    
    fprintf('─────────────────────────────────────────────────────────\n');
    fprintf('Testing: %s\n', base_name);
    fprintf('─────────────────────────────────────────────────────────\n');
    
    % Load info file
    info_file = ['test_data/' base_name '_info.txt'];
    if ~exist(info_file, 'file')
        fprintf('  Info file not found, skipping...\n\n');
        continue;
    end
    
    % Parse shape from info file
    fid = fopen(info_file, 'r');
    info_lines = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    info_text = info_lines{1}{1};  % First line contains shape
    
    % Extract shape
    shape_str = regexp(info_text, '\[(.*?)\]', 'tokens');
    shape = str2num(shape_str{1}{1});
    
    % Load binary weights
    bin_file = ['test_data/' filename];
    fid = fopen(bin_file, 'rb');
    weights = fread(fid, inf, 'float32');
    fclose(fid);
    
    % Reshape
    weights_tensor = reshape(weights, shape);
    
    fprintf('  Shape: [%s]\n', num2str(shape));
    fprintf('  Parameters: %d\n', numel(weights));
    fprintf('  Size: %.2f KB\n', numel(weights) * 4 / 1024);
    fprintf('\n');
    
    % TT-decomposition with different tolerances
    tolerances = [1e-2, 1e-3, 1e-4];
    
    fprintf('  TT-Decomposition Results:\n');
    fprintf('  %-12s %-20s %-15s %-15s %-10s\n', ...
        'Tolerance', 'Ranks', 'Storage', 'Compression', 'Error');
    fprintf('  %s\n', repmat('─', 1, 75));
    
    for tol_idx = 1:length(tolerances)
        tol = tolerances(tol_idx);
        
        % Decompose
        tt_weights = tt_tensor(weights_tensor, tol);
        
        % Compute error
        reconstructed = full(tt_weights);
        rel_error = norm(weights_tensor(:) - reconstructed(:)) / norm(weights_tensor(:));
        
        % Stats
        ranks = tt_weights.r;
        storage = numel(tt_weights.core);
        compression = numel(weights) / storage;
        
        fprintf('  %-12.0e [%-18s] %-15d %-15.2fx %-10.2e\n', ...
            tol, num2str(ranks'), storage, compression, rel_error);
        
        % Save for C++ comparison (only for 1e-3 tolerance)
        if tol == 1e-3
            % Save MATLAB result
            save(['test_data/matlab_result_' base_name '.mat'], ...
                'tt_weights', 'weights_tensor', 'shape');
            
            % Save cores
            for k = 1:length(ranks)-1
                core_k = core(tt_weights, k);
                filename_core = sprintf('test_data/matlab_core_%s_%d.bin', base_name, k);
                fid_core = fopen(filename_core, 'wb');
                if fid_core == -1
                    error(['Cannot create: ' filename_core]);
                end
                fwrite(fid_core, core_k(:), 'float32');
                fclose(fid_core);
            end
            
            fprintf('    └─ Saved for C++ comparison (tol=1e-3)\n');
            
            % Check if C++ results exist
            cpp_core_file = sprintf('test_data/cpp_core_%s_1.bin', base_name);
            if exist(cpp_core_file, 'file')
                fprintf('\n  C++ Implementation Verification:\n');
                try
                    % Load all C++ cores
                    cpp_cores = {};
                    cpp_ranks = [1];  % Start with rank 1
                    k = 1;
                    while exist(sprintf('test_data/cpp_core_%s_%d.bin', base_name, k), 'file')
                        fid = fopen(sprintf('test_data/cpp_core_%s_%d.bin', base_name, k), 'rb');
                        core_data = fread(fid, inf, 'float32');
                        fclose(fid);
                        
                        % Infer core dimensions from MATLAB reference
                        ref_core = core(tt_weights, k);
                        core_size = size(ref_core);
                        
                        if numel(core_data) == numel(ref_core)
                            cpp_cores{k} = reshape(core_data, core_size);
                            cpp_ranks(end+1) = core_size(3);
                        else
                            fprintf('    ⚠ Core %d size mismatch (expected %d, got %d)\n', ...
                                k, numel(ref_core), numel(core_data));
                            break;
                        end
                        k = k + 1;
                    end
                    
                    if ~isempty(cpp_cores)
                        % Reconstruct from C++ cores
                        cpp_tt = tt_tensor();
                        cpp_tt.d = length(cpp_cores);
                        cpp_tt.n = shape;
                        cpp_tt.r = cpp_ranks;
                        cpp_tt.core = cell2core(cpp_cores);
                        cpp_tt.ps = cumsum([1; cpp_ranks(1:end-1)' .* shape' .* cpp_ranks(2:end)']);
                        
                        cpp_reconstructed = full(cpp_tt);
                        cpp_error = norm(weights_tensor(:) - cpp_reconstructed(:)) / norm(weights_tensor(:));
                        cpp_storage = sum(cpp_ranks(1:end-1) .* shape .* cpp_ranks(2:end));
                        cpp_compression = numel(weights) / cpp_storage;
                        
                        fprintf('    Ranks: [%s]\n', num2str(cpp_ranks));
                        fprintf('    Storage: %d params\n', cpp_storage);
                        fprintf('    Compression: %.2f×', cpp_compression);
                        if cpp_compression < 1.0
                            fprintf(' ⚠️  EXPANSION!\n');
                        elseif cpp_compression < 1.5
                            fprintf(' (marginal)\n');
                        else
                            fprintf(' ✓\n');
                        end
                        fprintf('    Reconstruction error: %.2e', cpp_error);
                        if cpp_error < 1e-10
                            fprintf(' (near-perfect)\n');
                        elseif cpp_error < 1e-3
                            fprintf(' (very good)\n');
                        elseif cpp_error < 0.01
                            fprintf(' (acceptable)\n');
                        else
                            fprintf(' ⚠️  HIGH!\n');
                        end
                    end
                catch err
                    fprintf('    ⚠ Error loading C++ results: %s\n', err.message);
                end
            end
        end
    end
    
    fprintf('\n');
end

fprintf('═════════════════════════════════════════════════════════\n');
fprintf('Summary\n');
fprintf('═════════════════════════════════════════════════════════\n');
fprintf('TT-decomposition can compress CNN weights by 10-100×\n');
fprintf('Trade-off: Accuracy vs Compression\n');
fprintf('  • 1e-2: Aggressive compression, visible quality loss\n');
fprintf('  • 1e-3: Good balance for most applications\n');
fprintf('  • 1e-4: Conservative, minimal quality loss\n');
fprintf('\n');
fprintf('Next: Run C++ comparison program to compare implementations\n');

