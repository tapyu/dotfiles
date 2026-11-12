function C = iStringifyEmpties(~, C)
%   Copyright 2023 The MathWorks, Inc.

for i = 1:numel(C)
    if isempty(C{i})
        C{i} = "''";
    end
end
end

