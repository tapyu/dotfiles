function uniqueName = makeUniqueName(existingNames, Name)
%   Make new name(s) unique from a set of existing names. 

%   Copyright 2019-2020 The MathWorks, Inc.

if nargin > 1
    % Make a single name unique, returning a single value    
    uniqueNames = matlab.lang.makeUniqueStrings([existingNames, {Name}], numel(existingNames)+1, namelengthmax);
    uniqueName = uniqueNames{end};
    
else % nargin == 1
    % Make all of the existing names unique, returning a cell array of
    % unique names
    uniqueName = matlab.lang.makeUniqueStrings(existingNames);
end

end

