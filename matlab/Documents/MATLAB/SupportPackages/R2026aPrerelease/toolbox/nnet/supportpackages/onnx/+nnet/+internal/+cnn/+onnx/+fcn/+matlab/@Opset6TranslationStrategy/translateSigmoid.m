function nodeTranslation = translateSigmoid(this, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023 The MathWorks, Inc.

nodeTranslation = translateUnaryDlarrayOperator(this, nodeTranslation, node, 'sigmoid', IntegerTensorNames);
end

