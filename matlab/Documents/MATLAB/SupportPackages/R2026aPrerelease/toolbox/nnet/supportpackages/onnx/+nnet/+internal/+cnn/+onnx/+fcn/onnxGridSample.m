function [Y, numDimsY] = onnxGridSample(X, numDimsX, grid, align_corners, mode, padding_mode)
% ONNX GridSample operator
% 
% X: Input tensor in reverse ONNX order
%    2D: [W, H, C, N] (ONNX: [N, C, H, W])
% grid: Sampling coordinates in reverse ONNX order
%    2D: [2, W_out, H_out, N] where grid(1,:,:,:) = x, grid(2,:,:,:) = y

% Copyright 2024-2026 The MathWorks, Inc.

numDimsY = numDimsX;
[W, H, C, N] = size(X);
[~, W_out, H_out, ~] = size(grid);

% Extract normalized coordinates [-1, 1] from grid
norm_x = squeeze(grid(1,:,:,:)); % [W_out, H_out, N]
norm_y = squeeze(grid(2,:,:,:)); % [W_out, H_out, N]

pixel_x = GsDenormalize(norm_x, W, align_corners);
pixel_y = GsDenormalize(norm_y, H, align_corners);

% Perform interpolation with padding
switch mode
    case 'nearest'
        Y = interpolateNearest2D(X, pixel_x, pixel_y, W, H, C, N, W_out, H_out, padding_mode, align_corners);
    case 'bilinear'
        Y = interpolateBilinear2D(X, pixel_x, pixel_y, W, H, C, N, W_out, H_out, padding_mode, align_corners);
    case 'bicubic'
        Y = interpolateBicubic2D(X, pixel_x, pixel_y, W, H, C, N, W_out, H_out, padding_mode, align_corners);
end
end

function pixel_coord = GsDenormalize(norm_coord, size, align_corners)
    if align_corners
        pixel_coord = ((norm_coord + 1) / 2) * (size - 1) + 1;
    else
        pixel_coord = ((norm_coord + 1) * size - 1) / 2 + 1;
    end
end

function pix = PixelAtGrid2D(X, xi, yj, W, H, C, N, out_size, padding_mode, align_corners)
    % X: [W, H, C, N]
    % xi, yj: coordinates [W_out, H_out, N]
    % out_size: [W_out, H_out, C, N]
    % Returns: [W_out, H_out, C, N]
    
    [W_out, H_out, N_coords] = size(xi);
    
    % Expand coordinates to include channel dimension
    % xi, yj: [W_out, H_out, 1, N]
    xi = reshape(xi, [W_out, H_out, 1, N_coords]);
    yj = reshape(yj, [W_out, H_out, 1, N_coords]);

    % Preprocess for different padding_mode
    if isequal(padding_mode, 'reflection')
        if align_corners
            xi = GsReflect(xi, 1, W);
            yj = GsReflect(yj, 1, H);
        else
            xi = GsReflect(xi, 0.5, W + 0.5);
            yj = GsReflect(yj, 0.5, H + 0.5);
        end
    elseif isequal(padding_mode, 'zeros')
        % Check validity: [W_out, H_out, 1, N]
        valid = (xi >= 1) & (xi <= W) & (yj >= 1) & (yj <= H);
    end

    % Clamp for safe indexing
    xi = max(1, min(W, xi));
    yj = max(1, min(H, yj));

    % Compute base indices for spatial location: [W_out, H_out, 1, N]
    spatial_idx = sub2ind([W, H], xi, yj);
    % Expand to all channels: [W_out, H_out, C, N]
    channel_offsets = reshape(0:C-1, [1, 1, C, 1]) * W * H;
    batch_offsets = reshape(0:N-1, [1, 1, 1, N]) * W * H * C;

    switch padding_mode
        case 'zeros' 
            idx = spatial_idx + channel_offsets + batch_offsets;
            pix = reshape(X(idx), out_size) .* valid;
        case {'border', 'reflection'}
            idx = spatial_idx + channel_offsets + batch_offsets;
            pix = reshape(X(idx), out_size);
    end
end


function Y = interpolateNearest2D(X, pixel_x, pixel_y, W, H, C, N, W_out, H_out, padding_mode, align_corners)
    % Round to nearest (ONNX uses round-half-to-even)
    px = round(pixel_x);
    py = round(pixel_y);
    
    % Get all pixels at once for all channels and batches
    Y = PixelAtGrid2D(X, px, py, W, H, C, N, [W_out, H_out, C, N], padding_mode, align_corners);
end

function Y = interpolateBilinear2D(X, pixel_x, pixel_y, W, H, C, N, W_out, H_out, padding_mode, align_corners)
    % Get corner coordinates
    px_floor = floor(pixel_x);
    py_floor = floor(pixel_y);
    
    % Fractional parts: [W_out, H_out, N]
    wx = pixel_x - px_floor;
    wy = pixel_y - py_floor;
    
    % Expand weights to include channel dimension: [W_out, H_out, 1, N]
    wx = reshape(wx, [W_out, H_out, 1, N]);
    wy = reshape(wy, [W_out, H_out, 1, N]);
    
    % Four corners
    x0 = px_floor;
    y0 = py_floor;
    x1 = x0 + 1;
    y1 = y0 + 1;
    
    % Get all 4 corner pixels at once: [W_out, H_out, C, N]
    out_size = [W_out, H_out, C, N];
    p00 = PixelAtGrid2D(X, x0, y0, W, H, C, N, out_size, padding_mode, align_corners);
    p01 = PixelAtGrid2D(X, x0, y1, W, H, C, N, out_size, padding_mode, align_corners);
    p10 = PixelAtGrid2D(X, x1, y0, W, H, C, N, out_size, padding_mode, align_corners);
    p11 = PixelAtGrid2D(X, x1, y1, W, H, C, N, out_size, padding_mode, align_corners);
    
    % Bilinear interpolation - vectorized over all dimensions
    Y = (1-wx).*(1-wy).*p00 + (1-wx).*wy.*p01 + wx.*(1-wy).*p10 + wx.*wy.*p11;
end

function Y = interpolateBicubic2D(X, pixel_x, pixel_y, W, H, C, N, W_out, H_out, padding_mode, align_corners)
    % Get top-left of 4x4 grid
    px_floor = floor(pixel_x);
    py_floor = floor(pixel_y);
    
    % Fractional offset
    fx = pixel_x - px_floor;
    fy = pixel_y - py_floor;
    
    % Top-left corner of 4x4 neighborhood
    x_nw = px_floor - 1;
    y_nw = py_floor - 1;
    
    % Get cubic coefficients: [W_out, H_out, 4]
    coeffs_x = GsGetCubicCoeffs(fx);
    coeffs_y = GsGetCubicCoeffs(fy);
    
    % Expand coefficients to include channel and batch: [W_out, H_out, 1, N, 4]
    coeffs_x = reshape(coeffs_x, [W_out, H_out, 1, N, 4]);
    coeffs_y = reshape(coeffs_y, [W_out, H_out, 1, N, 4]);
    
    Y = zeros(W_out, H_out, C, N, 'like', X);
    
    % Loop over 4x4 grid (still vectorized over C and N)
    for i = 0:3
        for j = 0:3
            xi = x_nw + i;
            yj = y_nw + j;
            
            pix = PixelAtGrid2D(X, xi, yj, W, H, C, N, [W_out, H_out, C, N], padding_mode, align_corners);
            % pix: [W_out, H_out, C, N]
            % coeffs: [W_out, H_out, 1, 1, 4]
            Y = Y + coeffs_x(:,:,:,:,i+1) .* coeffs_y(:,:,:,:,j+1) .* pix;
        end
    end
end

function x = GsReflect(x, x_min, x_max)
    range = x_max - x_min;
    
    % Handle below minimum
    mask_low = x < x_min;
    if any(mask_low(:))
        dx = x_min - x(mask_low);
        n = floor(dx / range);
        r = dx - n * range;
        is_even = (mod(extractdata(n), 2) == 0);
        x(mask_low) = is_even .* (x_min + r) + (~is_even) .* (x_max - r);
    end
    
    % Handle above maximum
    mask_high = x > x_max;
    if any(mask_high(:))
        dx = x(mask_high) - x_max;
        n = floor(dx / range);
        r = dx - n * range;
        is_even = (mod(extractdata(n), 2) == 0);
        x(mask_high) = is_even .* (x_max - r) + (~is_even) .* (x_min + r);
    end
end

function coeffs = GsGetCubicCoeffs(x)
    cubic_alpha = -0.75;
    x_abs = abs(x);
    
    x_p1 = x_abs + 1;
    x_0 = x_abs;
    x_m1 = abs(1 - x_abs);
    x_m2 = abs(2 - x_abs);
    
    c1 = ((cubic_alpha * x_p1 - 5*cubic_alpha) .* x_p1 + 8*cubic_alpha) .* x_p1 - 4*cubic_alpha;
    c2 = ((cubic_alpha + 2) * x_0 - (cubic_alpha + 3)) .* x_0 .* x_0 + 1;
    c3 = ((cubic_alpha + 2) * x_m1 - (cubic_alpha + 3)) .* x_m1 .* x_m1 + 1;
    c4 = ((cubic_alpha * x_m2 - 5*cubic_alpha) .* x_m2 + 8*cubic_alpha) .* x_m2 - 4*cubic_alpha;
    
    coeffs = cat(3, c1, c2, c3, c4);
end
