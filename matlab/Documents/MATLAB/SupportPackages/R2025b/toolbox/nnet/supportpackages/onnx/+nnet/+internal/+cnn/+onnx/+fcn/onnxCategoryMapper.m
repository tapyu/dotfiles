function [Y, numDimsX] = onnxCategoryMapper(X, Cat2Int, Int2Cat, default_int64, default_string, numDimsX)
% Implements the ONNX CategoryMapper operator. 

% Copyright 2020 The MathWorks, Inc.

if isa(X, 'string')
    % Map strings to ints
%     X         = deblank(X);
    Y         = dlarray(zeros(size(X)));
    iskey     = isKey(Cat2Int.map, cellstr(X));
    Y(iskey)  = arrayfun(@(s)Cat2Int.map(s), X(iskey));
    Y(~iskey) = default_int64;
else
    % Map ints to strings
    Y         = strings(size(X));
    X         = extractdata(X);
    iskey     = arrayfun(@(x)isKey(Int2Cat.map, x), X);
    % NOTE: containers.Map always returns chars instead of strings. 
    Y(iskey)  = string(arrayfun(@(i)Int2Cat.map(i), X(iskey), 'UniformOutput', false));
    Y(~iskey) = default_string;
end
end
