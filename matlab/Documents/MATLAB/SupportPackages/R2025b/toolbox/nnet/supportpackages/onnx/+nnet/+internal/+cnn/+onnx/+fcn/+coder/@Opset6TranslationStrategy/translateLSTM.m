function nodeTranslation = translateLSTM(this, nodeTranslation, node, ~)
%   Copyright 2024-2026 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "activation_alpha"	"FLOATS"    true    0
    "activation_beta" 	"FLOATS"    true    0
    "activations"       "STRINGS"   true    ["sigmoid","tanh","tanh"]
    "clip"              "FLOAT"     true    []
    "direction"         "STRING"    true    "forward"
    "hidden_size"    	"INT"       false   []
    "input_forget"    	"INT"       true    0
    "output_sequence"   "INT"       true    0
    });
% Parse the attributes
[~, ~, activations, ~, direction, ~, ~, ~] = ...
    nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
if isequal(direction, "reverse")
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node,...
        message("nnet_cnn_onnx:onnx:NoReverseLSTM"));
    return;
end
iofAct = iTranslateActivationFunction(this, activations(1));
cellAct = iTranslateActivationFunction(this, activations(2));
hidAct = iTranslateActivationFunction(this, activations(3));
% Check that activation functions are valid
if ~ismember(iofAct, ["sigmoid", "hard-sigmoid"])
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node,...
        message("nnet_cnn_onnx:onnx:UnsupportedDlarrayLSTMGateActivation"));
    return;
end
if ~ismember(cellAct, ["tanh", "softsign", "relu"])
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node,...
        message("nnet_cnn_onnx:onnx:UnsupportedDlarrayLSTMCellActivation"));
    return;
end
if ~ismember(hidAct, ["tanh", "softsign", "relu"])
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node,...
        message("nnet_cnn_onnx:onnx:UnsupportedDlarrayLSTMHiddenActivation"));
    return;
end
if hidAct~=cellAct
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node,...
        message("nnet_cnn_onnx:onnx:CellAndHiddenActsUnequal"));
    return;
end
% [Y, HIDDENSTATE, CELLSTATE, rankY, rankH, rankC] = LSTM(X, W, R, B, sequence_lens, initial_h, initial_c, P)
Y           = '~';
HIDDENSTATE = '~';
CELLSTATE   = '~';
rankY       = '~';
rankH       = '~';
rankC       = '~';
X           = node.input{1};
W           = node.input{2};
R           = node.input{3};
[X, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
setInitializerType(nodeTranslation.GraphTranslation,'Learnables',W);
[W, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, W);
setInitializerType(nodeTranslation.GraphTranslation,'Learnables',R);
[R, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, R);
B           = "''";
sequence_lens = "''";
initial_h   = "''";
initial_c   = "''";
P           = "''";
% Process the 3 optional output arguments
if numel(node.output) > 0 && ~isempty(node.output{1})
    [Y, rankY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, node.output{1});
end
if numel(node.output) > 1 && ~isempty(node.output{2})
    [HIDDENSTATE, rankH] = getVarAndNumDimName(nodeTranslation.GraphTranslation, node.output{2});
end
if numel(node.output) > 2 && ~isempty(node.output{3})
    [CELLSTATE, rankC] = getVarAndNumDimName(nodeTranslation.GraphTranslation, node.output{3});
end
% Process the 5 optional input arguments
if numel(node.input) > 3 && ~isempty(node.input{4})
    setInitializerType(nodeTranslation.GraphTranslation,'Learnables',node.input{4});
    [B, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, node.input{4});
end
if numel(node.input) > 4 && ~isempty(node.input{5})
    [sequence_lens, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, node.input{5});
end
if numel(node.input) > 5 && ~isempty(node.input{6})
    [initial_h, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, node.input{6});
end
if numel(node.input) > 6 && ~isempty(node.input{7})
    [initial_c, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, node.input{7});
end
if numel(node.input) > 7 && ~isempty(node.input{8})
    [P, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, node.input{8});
end
InputArgs = iStringifyEmpties(this, {W, R, B, sequence_lens, initial_h, initial_c, P}); % See ONNX LSTM Doc for the meaning of these arguments.
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('nnet.internal.cnn.coder.onnx.codererror(true, ''nnet_cnn_onnx:onnx:UnsupportedOperatorForCodegen'',''%s'');\n', node.op_type),...   
    ];
% switch direction
%     case 'forward'
%         command = [
%             sprintf('%% %s:\n', node.op_type),...
%             sprintf('[H0, C0, WEIGHTS, RECURRENTWEIGHTS, BIAS, %s, %s, %s] = %sprepareLSTMArgs(%s, %s, %s, %s, %s, %s, %s);\n', ...
%             rankY, rankH, rankC, getPackageName(nodeTranslation.GraphTranslation), InputArgs{:}),...
%             sprintf('[%s, %s, %s] = lstm(%s, H0, C0, WEIGHTS, RECURRENTWEIGHTS, BIAS, ''DataFormat'', ''CBT'', ''StateActivationFunction'', ''%s'', ''GateActivationFunction'', ''%s'');\n', Y, HIDDENSTATE, CELLSTATE, X, hidAct, iofAct),...
%             ];
%         if Y~='~'
%             % Sequence output is requested. Add code to reshape it:
%             command = [command,...
%                 sprintf('[C, B, T] = size(%s, 1:3);\n', Y),...
%                 sprintf('%s = reshape(%s, [C B 1 T]);\n', Y, Y),...
%                 ];
%         end
%         nodeTranslation.IncludedFunctionNames   = "prepareLSTMArgs";
%     case 'bidirectional'
%         command = [
%             sprintf('%% %s:\n', node.op_type),...
%             sprintf('[%s, %s, %s, %s, %s, %s] = %sonnxBidirectionalLSTM(%s, %s, %s, %s, %s, %s, %s, %s, ''%s'', ''%s'');\n', ...
%             Y, HIDDENSTATE, CELLSTATE, rankY, rankH, rankC, getPackageName(nodeTranslation.GraphTranslation), X, InputArgs{:}, hidAct, iofAct),...
%             ];
%         nodeTranslation.IncludedFunctionNames = "onnxBidirectionalLSTM";
%     otherwise
%         assert(false);
% end
nodeTranslation.MCode                   = string(command);
end