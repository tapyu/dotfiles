function nodeTranslation = translateReduceMean(this, nodeTranslation, node, IntegerTensorNames)
    %   Copyright 2024 The MathWorks, Inc. 

    if ismember(node.input{1}, IntegerTensorNames)
        op = '%s = fix(mean(%s, dims));\n';
    else
        op = '%s = mean(%s, dims);\n';
    end
    nodeTranslation = translateReduceOp(this, nodeTranslation, node, IntegerTensorNames, op);
end