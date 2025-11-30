% Debug: Check tensor layout

fprintf('=== Tensor Layout Debugging ===\n\n');

% Load original tensor from MATLAB
load('test_data/matlab_result_small.mat', 'tensor_small');

fprintf('MATLAB tensor_small:\n');
disp(tensor_small);

% Load binary tensor (what C++ reads)
fid = fopen('test_data/tensor_small.bin', 'rb');
tensor_bin = fread(fid, inf, 'float32');
fclose(fid);

fprintf('Binary file (linear):\n');
fprintf('  [');
for i = 1:length(tensor_bin)
    fprintf('%.0f ', tensor_bin(i));
end
fprintf(']\n\n');

% Reshape binary as C++ would (row-major assumption)
fprintf('If C++ reads as [2,2,2] tensor:\n');
tensor_cpp_view = reshape(tensor_bin, [2,2,2]);
disp(tensor_cpp_view);

fprintf('C++ layout (assuming row-major semantics):\n');
fprintf('  T[0,0,0]=%g, T[0,0,1]=%g\n', tensor_cpp_view(1,1,1), tensor_cpp_view(1,1,2));
fprintf('  T[0,1,0]=%g, T[0,1,1]=%g\n', tensor_cpp_view(1,2,1), tensor_cpp_view(1,2,2));
fprintf('  T[1,0,0]=%g, T[1,0,1]=%g\n', tensor_cpp_view(2,1,1), tensor_cpp_view(2,1,2));
fprintf('  T[1,1,0]=%g, T[1,1,1]=%g\n', tensor_cpp_view(2,2,1), tensor_cpp_view(2,2,2));

fprintf('\nMATLAB layout (column-major):\n');
fprintf('  T(1,1,1)=%g, T(1,1,2)=%g\n', tensor_small(1,1,1), tensor_small(1,1,2));
fprintf('  T(1,2,1)=%g, T(1,2,2)=%g\n', tensor_small(1,2,1), tensor_small(1,2,2));
fprintf('  T(2,1,1)=%g, T(2,1,2)=%g\n', tensor_small(2,1,1), tensor_small(2,1,2));
fprintf('  T(2,2,1)=%g, T(2,2,2)=%g\n', tensor_small(2,2,1), tensor_small(2,2,2));

