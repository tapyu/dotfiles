function isValid = isValidDataPermutation(perm, varargin)

%   Copyright 2020 The MathWorks, Inc.    

isValid = isnumeric(perm) && numel(varargin) == 1 ...
    || isa(perm, 'cell') && numel(perm) == numel(varargin) ...
    || strcmp(perm, 'auto')...
    || strcmp(perm, 'none');
end
 