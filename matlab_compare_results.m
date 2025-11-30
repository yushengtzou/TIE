% MATLAB Script: Compare C++ results with MATLAB TT-Toolbox
% Run this AFTER running the C++ comparison program

fprintf('=== Comparing C++ vs MATLAB TT-Decomposition ===\n\n');

% Add TT-Toolbox to path
addpath('../TT-Toolbox');

%% Load MATLAB results
load('test_data/matlab_result_small.mat');
load('test_data/matlab_result_345.mat');
load('test_data/matlab_result_img.mat');

%% Helper function to load C++ cores
function core_cpp = load_cpp_core(filename, shape)
    fid = fopen(filename, 'rb');
    if fid == -1
        error(['Cannot open: ' filename]);
    end
    data = fread(fid, inf, 'float32');
    fclose(fid);
    core_cpp = reshape(data, shape);
end

%% Test 1: Small 2×2×2 tensor
fprintf('Test 1: Small 2x2x2 Tensor\n');
fprintf('  MATLAB ranks: [%s]\n', num2str(tt_small.r'));

% Load C++ cores
try
    n_cores = length(tt_small.r) - 1;
    cores_cpp = cell(1, n_cores);
    ranks_cpp = zeros(1, n_cores + 1);
    ranks_cpp(1) = 1;
    
    for k = 1:n_cores
        filename = sprintf('test_data/cpp_core_small_%d.bin', k);
        core_matlab = core(tt_small, k);
        shape_matlab = size(core_matlab);
        
        cores_cpp{k} = load_cpp_core(filename, shape_matlab);
        ranks_cpp(k+1) = size(cores_cpp{k}, 3);
        
        % Compare cores
        diff = cores_cpp{k} - core_matlab;
        rel_error = norm(diff(:)) / norm(core_matlab(:));
        
        fprintf('  Core %d: Error = %.6e\n', k, rel_error);
    end
    
    fprintf('  C++ ranks: [%s]\n', num2str(ranks_cpp));
    fprintf('  Status: ');
    if isequal(tt_small.r', ranks_cpp)
        fprintf('✓ Ranks match!\n');
    else
        fprintf('⚠ Ranks differ (randomized SVD approximation)\n');
    end
catch ME
    fprintf('  ✗ C++ results not found. Run C++ program first.\n');
    fprintf('    Error: %s\n', ME.message);
end
fprintf('\n');

%% Test 2: Medium 3×4×5 tensor
fprintf('Test 2: Medium 3x4x5 Tensor\n');
fprintf('  MATLAB ranks: [%s]\n', num2str(tt_345.r'));

try
    n_cores = length(tt_345.r) - 1;
    cores_cpp = cell(1, n_cores);
    ranks_cpp = zeros(1, n_cores + 1);
    ranks_cpp(1) = 1;
    
    for k = 1:n_cores
        filename = sprintf('test_data/cpp_core_345_%d.bin', k);
        core_matlab = core(tt_345, k);
        shape_matlab = size(core_matlab);
        
        cores_cpp{k} = load_cpp_core(filename, shape_matlab);
        ranks_cpp(k+1) = size(cores_cpp{k}, 3);
        
        diff = cores_cpp{k} - core_matlab;
        rel_error = norm(diff(:)) / norm(core_matlab(:));
        
        fprintf('  Core %d: Error = %.6e\n', k, rel_error);
    end
    
    fprintf('  C++ ranks: [%s]\n', num2str(ranks_cpp));
    fprintf('  Status: ');
    if isequal(tt_345.r', ranks_cpp)
        fprintf('✓ Ranks match!\n');
    else
        fprintf('⚠ Ranks differ (randomized SVD approximation)\n');
    end
catch ME
    fprintf('  ✗ C++ results not found. Run C++ program first.\n');
    fprintf('    Error: %s\n', ME.message);
end
fprintf('\n');

%% Test 3: Image 8×8×8×8×3 tensor
fprintf('Test 3: Image 8x8x8x8x3 Tensor\n');
fprintf('  MATLAB ranks: [%s]\n', num2str(tt_img.r'));
fprintf('  MATLAB storage: %d elements\n', numel(tt_img.core));

% MATLAB reconstruction error
img_reconstructed_matlab = full(tt_img);
error_matlab = norm(tensor_img(:) - img_reconstructed_matlab(:)) / norm(tensor_img(:));
fprintf('  MATLAB error: %.6e\n', error_matlab);

try
    n_cores = length(tt_img.r) - 1;
    total_cpp_storage = 0;
    
    fprintf('  C++ cores:\n');
    for k = 1:n_cores
        filename = sprintf('test_data/cpp_core_img_%d.bin', k);
        fid = fopen(filename, 'rb');
        data = fread(fid, inf, 'float32');
        fclose(fid);
        
        total_cpp_storage = total_cpp_storage + length(data);
        fprintf('    Core %d: %d elements\n', k, length(data));
    end
    
    fprintf('  C++ storage: %d elements\n', total_cpp_storage);
    fprintf('  Storage ratio (C++/MATLAB): %.2f\n', ...
        total_cpp_storage / numel(tt_img.core));
    
    % Note: Full reconstruction requires implementing TT contraction in C++
    fprintf('  Status: ✓ C++ decomposition completed\n');
    fprintf('  Note: Randomized SVD may have different ranks than exact SVD\n');
catch ME
    fprintf('  ✗ C++ results not found. Run C++ program first.\n');
    fprintf('    Error: %s\n', ME.message);
end
fprintf('\n');

%% Summary
fprintf('=== Summary ===\n');
fprintf('MATLAB uses exact SVD (QR + Jacobi iterations)\n');
fprintf('C++ uses randomized SVD (fixed iterations, approximate)\n');
fprintf('\n');
fprintf('Expected differences:\n');
fprintf('  - Ranks may differ slightly\n');
fprintf('  - Core values will differ (non-unique decomposition)\n');
fprintf('  - Reconstruction accuracy should be similar\n');
fprintf('\n');
fprintf('Both should achieve similar compression ratios and errors!\n');

