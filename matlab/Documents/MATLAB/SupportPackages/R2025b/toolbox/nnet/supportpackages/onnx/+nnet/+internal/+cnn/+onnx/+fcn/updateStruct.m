function s = updateStruct(s, t)
% Set all existing fields in s from fields in t, ignoring extra fields in
% t. 
%   Copyright 2020 The MathWorks, Inc.    

for name = transpose(fieldnames(s))
    s.(name{1}) = t.(name{1});
end
end
