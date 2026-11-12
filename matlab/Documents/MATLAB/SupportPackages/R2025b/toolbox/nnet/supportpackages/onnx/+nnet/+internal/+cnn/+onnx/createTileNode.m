function [TileNode, inits] = createTileNode(opset, name, input, output)
% A helper function to create a Tile operator of the specified opset
% version. 

%   Copyright 2022 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
TileNode = NodeProto;
TileNode.op_type   = 'Tile';
TileNode.name      = name;
TileNode.input     = input;
TileNode.output    = output;
inits                = [];
end