function [H0, C0, WEIGHTS, RECURRENTWEIGHTS, BIAS, numDimsY, numDimsH, numDimsC] = prepareLSTMArgs(W, R, B, sequence_lens, initial_h, initial_c, P)
% Prepares arguments for implementing the ONNX LSTM operator, forward mode
% only.

%   Copyright 2020-2022 The MathWorks, Inc.    

% All input tensors have reverse-ONNX dimension ordering.
% W has shape [input_size, 4*hidden_size, num_directions], with num_directions=1.
[input_size, hidden_size4, num_directions] = size(W);
hidden_size = hidden_size4/4;
% initial_h shape is [hidden_size, batch_size, num_directions]
% H0 should have shape [hidden_size, batch_size]
if isempty(initial_h)
    H0 = dlarray(zeros(hidden_size, 1, num_directions));
else
    H0 = initial_h;
end
% initial_c shape is [hidden_size, batch_size, num_directions]
% C0 should have shape [hidden_size, batch_size]
if isempty(initial_c)
    C0 = dlarray(zeros(hidden_size, 1, num_directions));
else
    C0 = initial_c;
end
% W has shape [input_size, 4*hidden_size, num_directions]. ONNX order is
% Input-Output-Forget-Cell (x2 if bidirectional). WEIGHTS must be a matrix
% of size 4*NumHiddenUnits-by-InputSize. MATLAB order is
% Input-Forget-Cell-Output (x2 if bidirectional).
iofcRows = reshape(1:4*hidden_size, [], 4);                      % Columns of this correspond to ONNX's iofc ordering.
ifcoRows = iofcRows(:, [1 3 4 2]);                               % Columns of this correspond to DLT's ifco ordering.
ifcoInd  = ifcoRows(:);                                          % A column vector containing DLT's ifco ordering.
WEIGHTS  = permute(W, [2 1 3]);                                 % WEIGHTS is [4*hidden_size, inputSize].
WEIGHTS  = WEIGHTS(ifcoInd, :);                                 % Rows of WEIGHTS are now in DLT's required ifco order.
% R has shape [hidden_size, 4*hidden_size, num_directions].
% RECURRENTWEIGHTS must be a matrix of size
% 4*NumHiddenUnits-by-NumHiddenUnits. Need to convert iofc to ifco as we
% did for WEIGHTS.
RECURRENTWEIGHTS  = permute(R, [2 1 3]);                        % RECURRENTWEIGHTS is [4*hidden_size, inputSize].
RECURRENTWEIGHTS  = RECURRENTWEIGHTS(ifcoInd, :);               % Rows of RECURRENTWEIGHTS are now in DLT's required ifco order.
% B has shape [8*hidden_size, num_directions]
% ONNX order is Input-Output-Forget-Cell (x2 if bidirectional)
% MATLAB order is Input-Forget-Cell-Output (x2 if bidirectional)
if isempty(B)
    BIAS = dlarray(zeros(4*hidden_size, num_directions));
else
    % B in ONNX is [1, 8*hidden_size], where the first half are the
    % forward biases and the second half are the recurrent biases.
    % In MATLAB these biases are combined.
    BIAS = B(1:4*hidden_size) + B(4*hidden_size+1:end);	% Combine forward and recurrent biases.
    BIAS = BIAS(ifcoInd);                               % Convert to DLT's required ifco order.
end
% X shape is [input_size, batch_size, seq_length]. Y will have the same
% dimension ordering as X. That means 'CBT'
% Label the input tensors
H0               = dlarray(stripdims(H0), 'CB');
C0               = dlarray(stripdims(C0), 'CB');
WEIGHTS          = dlarray(stripdims(WEIGHTS), 'CU');
RECURRENTWEIGHTS = dlarray(stripdims(RECURRENTWEIGHTS), 'CU');
BIAS             = dlarray(stripdims(BIAS(:)), 'C');
numDimsY = 4;
numDimsH = 3;
numDimsC = 3;
end
