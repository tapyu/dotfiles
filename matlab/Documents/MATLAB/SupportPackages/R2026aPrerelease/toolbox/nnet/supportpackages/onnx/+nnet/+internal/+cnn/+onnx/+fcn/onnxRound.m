function [Y, numDimsX] = onnxRound(X, numDimsX)
% Implements the ONNX Loop operator, which rounds halves to the nearest
% even integer. 

% Copyright 2020 The MathWorks, Inc.

Y = round(X);
eX = extractdata(X);
halves = mod(eX,.5)==0 & mod(eX,1)~=0;
Y(halves) = 2*round(X(halves)/2);
end
