function nodeTranslation = translateAcos(this, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2025 The MathWorks, Inc.

nodeTranslation = translateUnaryOperator(this, nodeTranslation, node, 'acos', IntegerTensorNames);
end