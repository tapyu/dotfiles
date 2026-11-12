function uint8vector = rawData(A)
% Reorders a MATLAB numeric multidimensional array into row-major order,
% and then typecasts it into a vector of uint8's.

%   Copyright 2018-2024 The MathWorks, Inc.    

assert(isnumeric(A));
rowMajorData = permute(A, fliplr(1:ndims(A)));
if isdlarray(rowMajorData)
    % typecast does not support dlarrays, only numeric arrays
    rowMajorData = extractdata(rowMajorData);
end
uint8vector = typecast(rowMajorData(:)', 'uint8');
end
