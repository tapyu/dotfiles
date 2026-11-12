function [Y, numDimsY] = onnxMin(XCell, numDimsXArray)

% Copyright 2020 The MathWorks, Inc.

% Elementwise min with implicit expansion
Y = XCell{1};
for i=2:numel(XCell)
    Y = min(Y, XCell{i});   % implicit expansion done here.
end
numDimsY = max(numDimsXArray);
end
