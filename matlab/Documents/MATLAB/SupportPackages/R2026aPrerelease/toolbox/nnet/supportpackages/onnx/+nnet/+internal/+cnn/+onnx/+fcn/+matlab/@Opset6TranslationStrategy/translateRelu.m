function nodeTranslation = translateRelu(this, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023 The MathWorks, Inc.

nodeTranslation = translateUnaryDlarrayOperator(this, nodeTranslation, node, 'relu', IntegerTensorNames);
end

