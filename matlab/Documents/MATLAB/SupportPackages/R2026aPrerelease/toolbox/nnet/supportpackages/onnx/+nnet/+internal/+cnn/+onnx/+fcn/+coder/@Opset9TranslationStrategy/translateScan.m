function nodeTranslation = translateScan(~, nodeTranslation, node, ~)
%   Copyright 2024-2025 The MathWorks, Inc.

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('nnet.internal.cnn.coder.onnx.codererror(true, ''nnet_cnn_onnx:onnx:UnsupportedOperatorForCodegen'',''%s'');\n', node.op_type),...
  ];

nodeTranslation.MCode                	= string(command);
end

