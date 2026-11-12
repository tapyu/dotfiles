function [Y, numDimsX] = onnxCategoryMapper(X_, Strings_, Ints_, default_int64, default_string, numDimsX_)
% Implements the ONNX CategoryMapper operator. 
%#codegen

% Copyright 2024 The MathWorks, Inc.

X         = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
Strings   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(Strings_);
Ints      = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(Ints_);
numDimsX  = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);

if isa(X, 'string')
    % Map strings to ints
    % Output is a dlarray of doubles
    Y         = zeros(size(X));
    strInt    = default_int64;
    for i = 1:numel(X)
        for j = 1:numel(Strings)
            if strcmp(X{i},Strings{j})
                strInt = Ints(j);
                break
            end
        end
        Y(i) = strInt;        
    end
else
    % Map ints to strings
    % Output is a cell array of chars
    Y         = cell(numel(X),1);
    intStr    = char(default_string);
    for i = 1:numel(X)
        for j = 1:numel(Ints)
            if X(i) == Ints(j)
                intStr = char(Strings{j});
                break
            end
        end        
        Y{i} = intStr;
    end    
end
end
