function codererror(condition, messageStr, varargin)
%#codegen
% Wrap call to the internal coder errorIf function. 
%   Copyright 2024 The MathWorks, Inc.    
    coder.internal.errorIf(condition, messageStr, varargin{:});
end