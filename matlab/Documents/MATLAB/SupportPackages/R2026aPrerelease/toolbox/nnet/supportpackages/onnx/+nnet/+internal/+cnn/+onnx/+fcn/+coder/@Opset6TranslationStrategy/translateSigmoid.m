function nodeTranslation = translateSigmoid(this, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024 The MathWorks, Inc.

nodeTranslation = translateUnaryDlarrayOperator(this, nodeTranslation, node, 'sigmoid', IntegerTensorNames);
end

