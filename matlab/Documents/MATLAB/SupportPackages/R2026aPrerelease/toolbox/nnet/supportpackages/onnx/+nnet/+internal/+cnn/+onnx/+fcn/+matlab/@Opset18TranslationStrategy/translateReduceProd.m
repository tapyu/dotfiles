function nodeTranslation = translateReduceProd(this, nodeTranslation, node, IntegerTensorNames)
    %   Copyright 2024 The MathWorks, Inc. 

    op = '%s = prod(%s, dims);\n';
    nodeTranslation = translateReduceOp(this, nodeTranslation, node, IntegerTensorNames, op);
end