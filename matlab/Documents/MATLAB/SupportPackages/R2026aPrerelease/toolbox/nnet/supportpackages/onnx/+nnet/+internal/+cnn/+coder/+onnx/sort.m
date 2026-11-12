function Y = sort(X)
%#codegen
% Wrap call to the internal coder isConst function and 
% call into compile-time extrinsic if possible. 
%   Copyright 2024 The MathWorks, Inc.   
coder.internal.prefer_const(X)
if(coder.internal.isConst(X))
    % Use Compile-time extrinsic
    Y = coder.const(feval('sort',X));
else
    % If X is not coder.const, call runtime sort
    Y = sort(X);
end    
end