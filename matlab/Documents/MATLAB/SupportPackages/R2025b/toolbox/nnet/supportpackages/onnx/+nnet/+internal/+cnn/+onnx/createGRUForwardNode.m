function [gruNode, inits] = createGRUForwardNode(opset, name, input, output, ...
    InputSize, hasSequenceOutput, ...
    iofAct, hidAct, inputWeights, recurrentWeights, bias, ...
    numHiddenUnits, ResetGateMode )
% Makes a GRU node, and initializers for inputs W, R and B
% containing the passed inputWeights, recurrentWeights, and bias.

%   Copyright 2022 The MathWorks, Inc. 

import nnet.internal.cnn.onnx.*
% Set inputs, outputs, and attributes:
gruNode            = NodeProto;
gruNode.op_type    = 'GRU';
gruNode.name       = name;
gruNode.input      = input;
gruNode.output     = output;
gruNode.attribute = [...
    makeAttributeProto('activations', 'STRINGS',     {iofAct, hidAct}),...
    makeAttributeProto('direction',   'STRING',      'forward'),...
    makeAttributeProto('hidden_size', 'INT',         numHiddenUnits),...
    makeAttributeProto('linear_before_reset', 'INT', int8(any(strcmp(ResetGateMode, ...
    ["after-multiplication", "recurrent-bias-after-multiplication"] ) ) ) ),...
    ];
if opset < 7
    gruNode.attribute(end+1) = makeAttributeProto('output_sequence', 'INT', single(hasSequenceOutput));
end

% Make initializers for W, R, B. Use the names passed in:

[~, WName, RName, BName, ~, HName] = input{:};

nH = numHiddenUnits;
[rInd, zInd, hInd] = nnet.internal.cnn.util.gruGateIndices(nH);
[forwardInd, ~] = nnet.internal.cnn.util.forwardBackwardSequenceIndices(nH, 'gru');
zrcForward = forwardInd([zInd, rInd, hInd]);

W               = inputWeights(zrcForward,:);
WTensor         = zeros(1, 3*nH, InputSize, 'single');
WTensor(1,:,:)  = W;
WInit           = makeTensorProtoOfType(WName, size(WTensor,1:3), WTensor, TensorProto_DataType.FLOAT);

R               = recurrentWeights(zrcForward,:);
RTensor         = zeros(1, 3*nH, nH, 'single');
RTensor(1,:,:)  = R;
RInit           = makeTensorProtoOfType(RName, size(RTensor,1:3), RTensor, TensorProto_DataType.FLOAT);

biasIndices     = zrcForward;
if strcmp(ResetGateMode, 'recurrent-bias-after-multiplication')
    biasIndices = [biasIndices, zrcForward+3*nH];
end
B              	= bias(biasIndices);
nB              = numel(B);
BTensor      	= zeros(1, 6*nH, 'single');
BTensor(1,1:nB)	= B;                      % "Recurrent biases" are left at 0.
BInit           = makeTensorProtoOfType(BName, size(BTensor,1:2), BTensor, TensorProto_DataType.FLOAT);

inits = [WInit, RInit, BInit];
end
