function nodeTranslation = translateReduceL1(this, nodeTranslation, node, IntegerTensorNames)
    %   Copyright 2024 The MathWorks, Inc. 

    op = '%s = sum(abs(%s), dims);\n';
    nodeTranslation = translateReduceOp(this, nodeTranslation, node, IntegerTensorNames, op);
end