function [Y, numDimsY] = onnxMin(XCell, numDimsXArray)
% Elementwise min with implicit expansion
%#codegen

% Copyright 2024 The MathWorks, Inc.

[numDimsY, maxNumDimsIdx] = max(numDimsXArray);
Y = XCell{maxNumDimsIdx}; 
for i=2:numel(XCell)
    if i ~= maxNumDimsIdx
        Y = min(Y, XCell{i});   % implicit expansion done here.
    end
end
end
