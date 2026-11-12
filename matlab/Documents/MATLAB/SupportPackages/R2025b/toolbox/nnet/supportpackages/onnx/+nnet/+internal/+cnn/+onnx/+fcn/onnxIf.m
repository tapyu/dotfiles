function varargout = onnxIf(cond, thenGraphFcn, elseGraphFcn, numOutputVars, Vars, NumDims, Training, state)
% Implements the ONNX If operator

% cond is a logical scalar. thenGraphFcn and elseGraphFcn are function
% handles. state is updated by the subgraphs. The subgraph output is a cell array
% of graph outputs, possibly empty if the graph produced no outputs,
% followed by state. 

% Copyright 2020 The MathWorks, Inc.

varargout = cell(1, numOutputVars*2 + 1);
if cond
    [varargout{:}] = thenGraphFcn(Vars, NumDims, Training, state);
else
    [varargout{:}] = elseGraphFcn(Vars, NumDims, Training, state);
end
% varargout is {graphOutput1, ..., graphOutputK, graphOutputNDims1,...graphOutputNDimsk, state}

end
