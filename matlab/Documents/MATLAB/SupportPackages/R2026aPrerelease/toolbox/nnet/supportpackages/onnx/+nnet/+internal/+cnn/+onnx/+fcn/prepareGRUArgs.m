function [H0, WEIGHTS, RECURRENTWEIGHTS, BIAS, numDimsY, numDimsH] = prepareGRUArgs(W, R, B, sequence_lens, initial_h, linear_before_reset)
% Prepares arguments for implementing the ONNX GRU operator, forward mode
% only.

%   Copyright 2021-2026 The MathWorks, Inc.

% All input tensors have reverse-ONNX dimension ordering.
% W has shape [input_size, 3*hidden_size, num_directions], with num_directions=1.
[~, hidden_size3] = size(W);
hidden_size = hidden_size3/3;
% initial_h shape is [hidden_size, batch_size, num_directions]
% H0 should have shape [hidden_size, batch_size]
if isempty(initial_h)
    H0 = dlarray(zeros(hidden_size, 1));
else
    H0 = initial_h;
end

% Defensive guard: newly generated code always passes linear_before_reset,
% but this default keeps the function callable if invoked without it.
if nargin < 6 || isempty(linear_before_reset)
    linear_before_reset = 1;
end

% ONNX order is ZRH
% MATLAB order is  RZH
zrhRows = reshape(1:3*hidden_size, [], 3);                  % Columns of this correspond to ONNX's zrh ordering.
rzhRows = zrhRows(:, [2 1 3]);                              % Columns of this correspond to DLT's rzh ordering.
rzhInd  = rzhRows(:);                                       % A column vector containing DLT's rzh ordering.
WEIGHTS     = W';                                           % WEIGHTS is now [3*hiddenSize, inputSize].
WEIGHTS     = WEIGHTS(rzhInd, :);                           % Rows of WEIGHTS are now in DLT's required rzh order.
RECURRENTWEIGHTS = R';                                      % RECURRENTWEIGHTS is now [3*hiddenSize, hiddenSize].
RECURRENTWEIGHTS = RECURRENTWEIGHTS(rzhInd, :);             % Rows of RECURRENTWEIGHTS are now in DLT's required rzh order.

% ONNX bias B has shape [6*hidden_size, 1] = [Wb_z, Wb_r, Wb_h, Rb_z, Rb_r, Rb_h]
if isempty(B)
    if linear_before_reset == 0
        BIAS = dlarray(zeros(3*hidden_size, 1));
    else
        BIAS = dlarray(zeros(6*hidden_size, 1));
    end
else
    if linear_before_reset == 0
        % linear_before_reset=0 ('before-multiplication'): All biases are
        % simply additive. Combine Wb and Rb per gate, in RZH order.
        BIAS = B(rzhInd) + B(rzhInd + 3*hidden_size);
    else
        % linear_before_reset=1 ('recurrent-bias-after-multiplication'):
        % Keep Wb and Rb separate. MATLAB gru() expects [Wb_RZH; Rb_RZH]
        % (6*hidden).
        BIAS = [B(rzhInd); B(rzhInd + 3*hidden_size)];
    end
    BIAS = BIAS(:);
end

% Label the input tensors
H0               = dlarray(stripdims(H0), 'CB');
WEIGHTS          = dlarray(stripdims(WEIGHTS), 'CU');
RECURRENTWEIGHTS = dlarray(stripdims(RECURRENTWEIGHTS), 'CU');
BIAS             = dlarray(stripdims(BIAS(:)), 'C');
numDimsY = 4;
numDimsH = 3;
end
