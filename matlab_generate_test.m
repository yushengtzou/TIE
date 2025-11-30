% MATLAB Script: Generate test tensor and decompose with TT-Toolbox
% This creates test data for comparing with C++ implementation

% Add TT-Toolbox to path
addpath('../TT-Toolbox');
cd ../TT-Toolbox
setup
cd ../tt_decompose_code

fprintf('=== TT-Toolbox Test Data Generator ===\n\n');

%% Create test_data directory if it doesn't exist
if ~exist('test_data', 'dir')
    mkdir('test_data');
    fprintf('Created test_data/ directory\n');
end

%% Test Case 1: Small 2×2×2 tensor (same as C++ test)
fprintf('Test 1: Small 2x2x2 tensor\n');
tensor_small = [
    1, 2,
    3, 4,
    5, 6,
    7, 8
];
tensor_small = reshape(tensor_small, [2, 2, 2]);

% Save as binary (float)
fid = fopen('test_data/tensor_small.bin', 'wb');
if fid == -1
    error('Cannot create file: test_data/tensor_small.bin');
end
fwrite(fid, tensor_small(:), 'float32');
fclose(fid);

% TT decomposition with MATLAB
tt_small = tt_tensor(tensor_small, 1e-6);
fprintf('  Shape: [%s]\n', num2str(size(tensor_small)));
fprintf('  TT-ranks: [%s]\n', num2str(tt_small.r'));
fprintf('  Storage: %d elements\n', numel(tt_small.core));

% Save MATLAB results
save('test_data/matlab_result_small.mat', 'tt_small', 'tensor_small');

% Extract and save cores
for k = 1:length(tt_small.r)-1
    core_k = core(tt_small, k);
    filename = sprintf('test_data/matlab_core_small_%d.bin', k);
    fid = fopen(filename, 'wb');
    if fid == -1
        error(['Cannot create file: ' filename]);
    end
    fwrite(fid, core_k(:), 'float32');
    fclose(fid);
    fprintf('  Core %d: [%d × %d × %d]\n', k, size(core_k));
end
fprintf('\n');

%% Test Case 2: 3×4×5 tensor
fprintf('Test 2: 3x4x5 tensor\n');
rng(42);  % Fixed seed for reproducibility
tensor_345 = zeros(3, 4, 5);
for i = 1:3
    for j = 1:4
        for k = 1:5
            tensor_345(i, j, k) = (i + 1) * (j + 1) * (k + 1);
        end
    end
end

% Save as binary
fid = fopen('test_data/tensor_345.bin', 'wb');
if fid == -1
    error('Cannot create file: test_data/tensor_345.bin');
end
fwrite(fid, tensor_345(:), 'float32');
fclose(fid);

% TT decomposition
tt_345 = tt_tensor(tensor_345, 1e-6);
fprintf('  Shape: [%s]\n', num2str(size(tensor_345)));
fprintf('  TT-ranks: [%s]\n', num2str(tt_345.r'));
fprintf('  Storage: %d elements\n', numel(tt_345.core));

% Save results
save('test_data/matlab_result_345.mat', 'tt_345', 'tensor_345');

for k = 1:length(tt_345.r)-1
    core_k = core(tt_345, k);
    filename = sprintf('test_data/matlab_core_345_%d.bin', k);
    fid = fopen(filename, 'wb');
    if fid == -1
        error(['Cannot create file: ' filename]);
    end
    fwrite(fid, core_k(:), 'float32');
    fclose(fid);
    fprintf('  Core %d: [%d × %d × %d]\n', k, size(core_k));
end
fprintf('\n');

%% Test Case 3: Image-like tensor (8×8×8×8×3)
fprintf('Test 3: Image-like tensor (8x8x8x8x3)\n');
rng(123);
img_3d = rand(64, 64, 3);
tensor_img = reshape(img_3d, [8, 8, 8, 8, 3]);

% Save as binary
fid = fopen('test_data/tensor_img.bin', 'wb');
if fid == -1
    error('Cannot create file: test_data/tensor_img.bin');
end
fwrite(fid, tensor_img(:), 'float32');
fclose(fid);

% TT decomposition with different tolerance
tt_img = tt_tensor(tensor_img, 1e-3);
fprintf('  Shape: [%s]\n', num2str(size(tensor_img)));
fprintf('  TT-ranks: [%s]\n', num2str(tt_img.r'));
fprintf('  Storage: %d elements (original: %d)\n', ...
    numel(tt_img.core), numel(tensor_img));
fprintf('  Compression ratio: %.2fx\n', ...
    numel(tensor_img) / numel(tt_img.core));

% Reconstruction error
img_reconstructed = full(tt_img);
relative_error = norm(tensor_img(:) - img_reconstructed(:)) / norm(tensor_img(:));
fprintf('  Relative error: %.6e\n', relative_error);

% Save results
save('test_data/matlab_result_img.mat', 'tt_img', 'tensor_img');

for k = 1:length(tt_img.r)-1
    core_k = core(tt_img, k);
    filename = sprintf('test_data/matlab_core_img_%d.bin', k);
    fid = fopen(filename, 'wb');
    if fid == -1
        error(['Cannot create file: ' filename]);
    end
    fwrite(fid, core_k(:), 'float32');
    fclose(fid);
    fprintf('  Core %d: [%d × %d × %d]\n', k, size(core_k));
end
fprintf('\n');

%% Save metadata as JSON-like text
fid = fopen('test_data/test_metadata.txt', 'w');
if fid == -1
    error('Cannot create file: test_data/test_metadata.txt');
end
fprintf(fid, 'Test Case 1: Small\n');
fprintf(fid, '  Shape: 2 2 2\n');
fprintf(fid, '  Ranks: %s\n', num2str(tt_small.r'));
fprintf(fid, '\n');
fprintf(fid, 'Test Case 2: Medium\n');
fprintf(fid, '  Shape: 3 4 5\n');
fprintf(fid, '  Ranks: %s\n', num2str(tt_345.r'));
fprintf(fid, '\n');
fprintf(fid, 'Test Case 3: Image\n');
fprintf(fid, '  Shape: 8 8 8 8 3\n');
fprintf(fid, '  Ranks: %s\n', num2str(tt_img.r'));
fprintf(fid, '  Tolerance: 1e-3\n');
fclose(fid);

fprintf('=== Test data generated successfully ===\n');
fprintf('Files saved in: test_data/\n');

