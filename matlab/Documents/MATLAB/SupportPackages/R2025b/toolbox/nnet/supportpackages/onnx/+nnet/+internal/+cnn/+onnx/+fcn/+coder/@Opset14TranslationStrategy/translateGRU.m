function nodeTranslation = translateGRU(this, nodeTranslation, node, ~)
%   Copyright 2024-2026 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "activation_alpha"	  "FLOATS"    true    0
    "activation_beta" 	  "FLOATS"    true    0
    "activations"         "STRINGS"   true    []
    "clip"                "FLOAT"     true    []
    "direction"           "STRING"    true    "forward"
    "hidden_size"    	  "INT"       false   []
    "linear_before_reset" "INT"       true    0
    "output_sequence"     "INT"       true    0
    "layout"              "INT"       true    0
    });
% Parse the attributes
[~, ~, activations, clip, direction, ~, linear_before_reset, ~, layout] = ...
    nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);

if isempty(activations)
    zrAct  = 'sigmoid';
    hidAct = 'tanh';
else
    zrAct  = iTranslateActivationFunction(this, activations{1});
    hidAct = iTranslateActivationFunction(this, activations{2});
end

% Check that attributes are valid
if ~ismember(zrAct, ["sigmoid", "hard-sigmoid"])
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node,...
        message('nnet_cnn_onnx:onnx:UnsupportedDlarrayLSTMGateActivation'));
    return;
end
if ~ismember(hidAct, ["tanh", "softsign"])
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node,...
        message('nnet_cnn_onnx:onnx:UnsupportedDlarrayLSTMHiddenActivation'));
    return;
end
if linear_before_reset ~= 1
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node,...
        message('nnet_cnn_onnx:onnx:UnsupportedAttributeValue', linear_before_reset, 'linear_before_reset'));
    return;
end
if ~isempty(clip)
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node,...
        message('nnet_cnn_onnx:onnx:UnsupportedAttribute', 'clip'));
    return;
end
if direction ~= "forward"
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node,...
        message("nnet_cnn_onnx:onnx:UnsupportedGRUDirection"));
    return;
end

inputDataFormat = 'CBT';
if layout==1
    % layout 1 is not currently supported in any third party libraries.
    % inputDataFormat = 'CTB'; % reverse-ONNX
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node,...
        message('nnet_cnn_onnx:onnx:UnsupportedAttributeValue', layout, 'layout'));
    return;
end

% [Y, HIDDENSTATE, rankY, rankH] = GRU(X, W, R, B, sequence_lens, initial_h)
Y           = '~';
HIDDENSTATE = '~';
rankY       = '~';
rankH       = '~';
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
% Process the 2 optional output arguments
if numel(node.output) > 0 && ~isempty(node.output{1})
    [Y, rankY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, node.output{1});
end
if numel(node.output) > 1 && ~isempty(node.output{2})
    [HIDDENSTATE, rankH] = getVarAndNumDimName(nodeTranslation.GraphTranslation, node.output{2});
end
% Process the 3 optional input arguments
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

InputArgs = iStringifyEmpties(this, {W, R, B, sequence_lens, initial_h}); % See ONNX GRU Doc for the meaning of these arguments.

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('nnet.internal.cnn.coder.onnx.codererror(true, ''nnet_cnn_onnx:onnx:UnsupportedOperatorForCodegen'',''%s'');\n', node.op_type),...    
    % sprintf('[H0, WEIGHTS, RECURRENTWEIGHTS, BIAS, %s, %s] = %sprepareGRUArgs(%s, %s, %s, %s, %s);\n', ...
    % rankY, rankH, getPackageName(nodeTranslation.GraphTranslation), InputArgs{:}),...
    % sprintf('[%s, %s] = gru(%s, H0, WEIGHTS, RECURRENTWEIGHTS, BIAS, ''DataFormat'', ''%s'', ''StateActivationFunction'', ''%s'', ''GateActivationFunction'', ''%s'');\n', Y, HIDDENSTATE, X, inputDataFormat, hidAct, zrAct),...
    ];
% if Y~='~'
%     % Sequence output is requested. Add code to reshape it:
% 
%     % The layout 1 is not yet supported.
%     %                 if layout==0
%     command = [command,...
%         sprintf('[C, B, T] = size(%s, 1:3);\n', Y),...
%         sprintf('%s = reshape(%s, [C B 1 T]);\n', Y, Y),...
%         ];
%     %                 else
%     %                     command = [command,...
%     %                         sprintf('[C, T, B] = size(%s, 1:3);\n', Y),...
%     %                         sprintf('%s = reshape(%s, [C 1 T B]);\n', Y, Y),...
%     %                     ];
%     %                 end
% end
nodeTranslation.MCode                   = string(command);
end