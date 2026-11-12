function [Y, numDimsY] = onnxMax(XCell, numDimsXArray)
% Elementwise max with implicit expansion
%#codegen

% Copyright 2024 The MathWorks, Inc.

[numDimsY, maxNumDimsIdx] = max(numDimsXArray);
Y = XCell{maxNumDimsIdx}; 
for i=2:numel(XCell)
    if i ~= maxNumDimsIdx
        Y = max(Y, XCell{i});   % implicit expansion done here.
    end
end
end
