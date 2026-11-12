function perm = preparePermutationVector(fromDimOrder, toDimOrder)
%preparePermutationVector Find the permutation vector that permutes one
%dimension ordering into another.
%
% Perm = preparePermutationVector(FromDimOrder, ToDimOrder) returns a vector
%     Perm that can be used to permute array dimensions from FromDimOrder
%     to ToDimOrder. If array A has dimension ordering FromDimOrder and
%     B=permute(A,perm), then B has dimension ordering ToDimOrder.
%
%   FromDimOrder and ToDimOrder must both be character vectors, string
%   scalars, string arrays, cell arrays of character vectors. The elements 
%   of FromDimOrder and ToDimOrder are treated as dimension labels.
%
% Example:
%   A = rand(2,3,4);
%   perm = preparePermutationVector('abc', 'bca')  
%   B = permute(A, perm);
%   sizeB = size(B)
%
% Example:
%   H = 100;
%   W = 200;
%   C = 3;
%   N = 64;
%   A = rand(H,W,C,N);
%   perm = preparePermutationVector('hwcn', 'nchw')  
%   B = permute(A, perm);
%   sizeA = size(A)
%   sizeB = size(B)
%
% Example:
%   A = rand(2,3,4);
%   perm = preparePermutationVector(["seq" "batch" "channel"], ["channel" "batch" "seq"])   

% Copyright 2020 The MathWorks, Inc.

% Parse the inputs
p = inputParser;
p.addRequired('fromDimOrder', @(s) validateattributes(s, ["string", "char"], "vector"));
p.addRequired('toDimOrder', @(s) validateattributes(s, ["string", "char"], "vector"));
parse(p,fromDimOrder, toDimOrder);

% Ensure both inputs are of the same type
if ~strcmp(class(fromDimOrder), class(toDimOrder))
    error(message('nnet_cnn_onnx:onnx:FPVtypes'));
end

% Ensure both inputs are of the same length
if numel(fromDimOrder) ~= numel(toDimOrder)
    error(message('nnet_cnn_onnx:onnx:FPVtypes'));
end

% If the inputs are string scalars, convert them to char vectors
if isstring(fromDimOrder) && isscalar(fromDimOrder)
    fromDimOrder = char(fromDimOrder);
end
if isstring(toDimOrder) && isscalar(toDimOrder)
    toDimOrder = char(toDimOrder);
end

% Use unique to get the sorted order of the elements in fromDimOrder and
% toDimOrder
[fromSorted, ifrom] = unique(fromDimOrder);
[toSorted, ~, iToInv] = unique(toDimOrder);

% If the number of unique elements is less than the number of original
% elements, the original dim order contains duplicates
if numel(fromSorted) ~= numel(fromDimOrder)
    error(message('nnet_cnn_onnx:onnx:FPVfromunique'));
end
if numel(toSorted) ~= numel(toDimOrder)
    error(message('nnet_cnn_onnx:onnx:FPVtounique'));
end

% If the unique sorted elements in 'from' and 'to' are not the same, they 
% do not contain the same elements.
if ~isequal(fromSorted, toSorted)
    error(message('nnet_cnn_onnx:onnx:FPVsame'));
end

% Get the permutation vector by using the indices to the sorted unique 
% elements in toDimOrder to index the indices of the original elements in 
% fromDimOrder.
perm = ifrom(iToInv);
perm = perm(:)';
end
