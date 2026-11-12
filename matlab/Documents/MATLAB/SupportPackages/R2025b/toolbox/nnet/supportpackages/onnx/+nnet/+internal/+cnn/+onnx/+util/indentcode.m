function chars = indentcode(chars)
% Wrap a call to the internal indentcode function. 
%   Copyright 2023 The MathWorks, Inc.    

try
    chars = indentcode(chars);
catch
    % Do nothing if it failed.
end
end