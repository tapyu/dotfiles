function [lstmNode, inits] = createLSTMForwardNode(opset, name, input, output, ...
    InputSize, hasSequenceOutput, ...
    iofAct, cellAct, hidAct, inputWeights, recurrentWeights, bias, ...
    numHiddenUnits )
% Makes a 'forward' mode LSTM node, and initializers for inputs W, R and B
% containing the passed inputWeights, recurrentWeights, and bias.

%   Copyright 2022 The MathWorks, Inc. 

import nnet.internal.cnn.onnx.*
% Set inputs, outputs, and attributes:
lstmNode            = NodeProto;
lstmNode.op_type    = 'LSTM';
lstmNode.name       = name;
lstmNode.input      = input;
lstmNode.output     = output;
lstmNode.attribute = [...
    makeAttributeProto('activations', 'STRINGS', {iofAct, cellAct, hidAct}),...
    makeAttributeProto('direction',   'STRING',  'forward'),...
    makeAttributeProto('hidden_size', 'INT',     numHiddenUnits),...
    ];
if opset < 7
    lstmNode.attribute(end+1) = makeAttributeProto('output_sequence', 'INT', single(hasSequenceOutput));
end

% Make initializers for W, R, B. Use the names passed in:

[~, WName, RName, BName, ~, HName, CName] = input{:};

nH = numHiddenUnits;
[cInd, iInd, fInd, oInd] = nnet.internal.cnn.util.gateIndices(nH);
[forwardInd, ~] = nnet.internal.cnn.util.forwardBackwardSequenceIndices(nH, 'lstm');
iofcForward = forwardInd([iInd oInd fInd cInd]);

W               = inputWeights(iofcForward,:);
WTensor         = zeros(1, 4*nH, InputSize, 'single');
WTensor(1,:,:)  = W;
WInit           = makeTensorProtoOfType(WName, size(WTensor,1:3), WTensor, TensorProto_DataType.FLOAT);

R               = recurrentWeights(iofcForward,:);
RTensor         = zeros(1, 4*nH, nH, 'single');
RTensor(1,:,:)  = R;
RInit           = makeTensorProtoOfType(RName, size(RTensor,1:3), RTensor, TensorProto_DataType.FLOAT);

B              	= bias(iofcForward);
nB              = numel(B);
BTensor      	= zeros(1, 8*nH, 'single');
BTensor(1,1:nB)	= B;                                                        % Fill from B, but "Recurrent biases" are left at 0.
BInit           = makeTensorProtoOfType(BName, size(BTensor,1:2), BTensor, TensorProto_DataType.FLOAT);

inits = [WInit, RInit, BInit];
end
