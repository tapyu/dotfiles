function v = prependOnes(sz, numDims)
% Takes a MATLAB size vector, prepends it with ones to numDims and converts it to
% int64. 
%   Copyright 2019 The MathWorks, Inc.    

assert(numel(sz) <= int64(numDims));
v = int64([ones(1,numDims-numel(sz)) sz]);
end
