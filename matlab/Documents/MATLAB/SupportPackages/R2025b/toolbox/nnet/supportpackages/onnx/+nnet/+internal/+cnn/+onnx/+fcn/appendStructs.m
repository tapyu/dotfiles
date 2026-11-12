function s = appendStructs(varargin)
% s = appendStructs(s1, s2,...). Assign all fields in s1, s2,... into s.

%   Copyright 2020 The MathWorks, Inc.    

if isempty(varargin)
    s = struct;
else
    s = varargin{1};
    for i = 2:numel(varargin)
        fromstr = varargin{i};
        fs = fieldnames(fromstr);
        for j = 1:numel(fs)
            s.(fs{j}) = fromstr.(fs{j});
        end
    end
end
end
