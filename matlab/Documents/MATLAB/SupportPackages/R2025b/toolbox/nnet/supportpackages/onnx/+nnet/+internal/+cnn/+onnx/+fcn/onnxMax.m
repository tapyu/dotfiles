function [Y, numDimsY] = onnxMax(XCell, numDimsXArray)

% Copyright 2020 The MathWorks, Inc.

% Elementwise max with implicit expansion
Y = XCell{1};
for i=2:numel(XCell)
    Y = max(Y, XCell{i});   % implicit expansion done here.
end
numDimsY = max(numDimsXArray);
end
