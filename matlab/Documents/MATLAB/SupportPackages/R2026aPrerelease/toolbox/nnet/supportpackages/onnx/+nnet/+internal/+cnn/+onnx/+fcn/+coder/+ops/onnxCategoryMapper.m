function [Y, numDimsX] = onnxCategoryMapper(X_, Strings_, Ints_, default_int64, default_string, numDimsX_)
% Implements the ONNX CategoryMapper operator. 
%#codegen

% Copyright 2024-2025 The MathWorks, Inc.

X         = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
Strings   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(Strings_);
Ints      = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(Ints_);
numDimsX  = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);

if iscell(X) || isa(X, 'string')
    % Map strings to ints
    % Output is an dlarray of doubles
    Y = repmat(default_int64, size(X));
    for i = 1:numel(X)
        for j = 1:numel(Strings)
            if strcmp(X{i},Strings{j})
                Y(i) = Ints(j);
                break
            end
        end
    end
else
    % Map ints to strings
    % Output is a cell array of chars
    intStr = char(default_string);
    Y      = repmat({intStr}, size(X));
    for i = 1:numel(X)
        for j = 1:numel(Ints)
            if X(i) == Ints(j)
                Y{i} = char(Strings{j});
                break
            end
        end        
    end    
end
end
