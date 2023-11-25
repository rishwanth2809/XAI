clc
clear all
close all


%% Problem definition



n = 1000;

% Generate data using LHS
Sw = 150 + (200-150) * lhsdesign(n, 1);
Wfw = 220 + (300-220) * lhsdesign(n, 1);
A = 6 + (10-6) * lhsdesign(n, 1);
delta = deg2rad(-10 + (20) * lhsdesign(n, 1));  % Convert degrees to radians
q = 16 + (45-16) * lhsdesign(n, 1);
lambda = 0.5 + (1-0.5) * lhsdesign(n, 1);
tc = 0.08 + (0.18-0.08) * lhsdesign(n, 1);
Nz = 2.5 + (6-2.5) * lhsdesign(n, 1);
Wdg = 1700 + (2500-1700) * lhsdesign(n, 1);
Wp = 0.025 + (0.08-0.025) * lhsdesign(n, 1);

% Calculate the wing weight output for each data point
y = zeros(n, 1);
for i = 1:n
    y(i) = wing_weight(Sw(i), Wfw(i), A(i), delta(i), q(i), lambda(i), tc(i), Nz(i), Wdg(i), Wp(i));
end

% Concatenate into a data matrix
data = [Sw Wfw A delta q lambda tc Nz Wdg Wp y];

% Display first few rows for verification
% disp(data(1:5, :));

% function fx = wing_weight(Sw, Wfw, A, delta, q, lambda, tc, Nz, Wdg, Wp)
%     % Wing weight function based on the provided formula
%     fx = 0.0368 * Sw^0.758 * Wfw^0.0035 ...
%         * (A/cos(delta)^2)^0.6 * q^0.006 * lambda^0.04 ...
%         * (100 * tc / cos(delta))^-0.3 ...
%         * (Nz * Wdg)^0.49 ...
%         + Sw * Wp;
% end

% Assuming your data is in a matrix 'data'

% Perform PCA on your data
[coeff, score, latent, tsquared, explained] = pca(data(:,1:10));

% Display explained variance
disp('Explained Variance Ratios:');
disp(explained(1:2));

% Display factor loadings for the first two principal components
disp('Factor Loadings (First two components):');
disp(coeff(:,1:2));

%% Dataset

% Define component names
comp_names = {'Sw', 'Wfw', 'A', 'delta', 'q', 'lambda', 'tc', 'Nz', 'Wdg', 'Wp', 'y'};

% Create SOM dataset and normalize
sData = som_data_struct(data,'comp_names',comp_names); 
sData = som_normalize(sData,'range');

%% Initializing SOM Map Codebook Vectors (Linear Initialization)
[sMap]= modifiedsom_lininit1(sData,'lattice','hexa','msize',[20,20]);

%% Training SOM
[sMap,sTrain] = modifiedsom_batchtrain(sMap,sData,'sample_order','ordered','trainlen',500,...
     'radius_ini', 1.0, 'radius_fin',0.9);

%% Denormalizing the data
sMap=som_denormalize(sMap,sData);   
sData=som_denormalize(sData,'remove');

%% Visualization of SOM results (U Matrix and Component Planes)
figure(3) 
som_show(sMap, 'umat','all','comp', 'all');

%% iSOM Grid in function space (using just the first three components for visualization)
figure(4)
som_grid(sMap,'coord',sMap.codebook(:,[1 2 11]),'label',sMap.labels,'labelcolor','c','labelsize',10, 'marker','o','MarkerColor','k',...
    'MarkerSize',7,'linecolor', 'k');
hold on, scatter3(data(:,1),data(:,2),data(:,11),20,'ro','filled');
xlabel('Sw')
ylabel('Wfw')
zlabel('y')

% Display explained variance
disp('Explained Variance Ratios:');
disp(explained(1:2));

% Display factor loadings for the first two principal components
disp('Factor Loadings (First two components):');
disp(coeff(:,1:2));

% % Perform PCA on your data
% [coeff, score, latent, tsquared, explained] = pca(sMap.codebook);
% 
% % Display explained variance
% disp('Explained Variance Ratios:');
% disp(explained(1:2));
% 
% % Display factor loadings for the first two principal components
% disp('Factor Loadings (First two components):');
% disp(coeff(:,1:2));
%% Cosine similarity
% Extracting the codebook vectors
v1 = sMap.codebook(:,1); % Sw
v2 = sMap.codebook(:,2); % Wfw
v3 = sMap.codebook(:,3); % A
v4 = sMap.codebook(:,4); % delta
v5 = sMap.codebook(:,5); % q
v6 = sMap.codebook(:,6); % lambda
v7 = sMap.codebook(:,7); % tc
v8 = sMap.codebook(:,8); % Nz
v9 = sMap.codebook(:,9); % Wdg
v10 = sMap.codebook(:,10); % Wp
v11 = sMap.codebook(:,11); % y (output)

% Compute cosine similarity for each variable with v11
similarities = zeros(10, 1);
similarities(1) = matrix_cosine_similarity(reshape(v1, [20, 20]), reshape(v11, [20, 20]));
similarities(2) = matrix_cosine_similarity(reshape(v2, [20, 20]), reshape(v11, [20, 20]));
similarities(3) = matrix_cosine_similarity(reshape(v3, [20, 20]), reshape(v11, [20, 20]));
similarities(4) = matrix_cosine_similarity(reshape(v4, [20, 20]), reshape(v11, [20, 20]));
similarities(5) = matrix_cosine_similarity(reshape(v5, [20, 20]), reshape(v11, [20, 20]));
similarities(6) = matrix_cosine_similarity(reshape(v6, [20, 20]), reshape(v11, [20, 20]));
similarities(7) = matrix_cosine_similarity(reshape(v7, [20, 20]), reshape(v11, [20, 20]));
similarities(8) = matrix_cosine_similarity(reshape(v8, [20, 20]), reshape(v11, [20, 20]));
similarities(9) = matrix_cosine_similarity(reshape(v9, [20, 20]), reshape(v11, [20, 20]));
similarities(10) = matrix_cosine_similarity(reshape(v10, [20, 20]), reshape(v11, [20, 20]));

% Bar plot
figure;
bar(similarities);
xlabel('Variables');
ylabel('Cosine Similarity with output');
title('Cosine Similarity of each variable with the output');
set(gca, 'XTick', 1:10, 'XTickLabel', {'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'});

% Compute slopes for each variable
Sw_slope = calculate_slope(v1);
Wfw_slope = calculate_slope(v2);
A_slope = calculate_slope(v3);
delta_slope = calculate_slope(v4);
q_slope = calculate_slope(v5);
lambda_slope = calculate_slope(v6);
tc_slope = calculate_slope(v7);
Nz_slope = calculate_slope(v8);
Wdg_slope = calculate_slope(v9);
Wp_slope = calculate_slope(v10);

% Compute the projected y slopes for each variable
Sw_y_slope_projected = calculate_y_slope(v11, Sw_slope);
Wfw_y_slope_projected = calculate_y_slope(v11, Wfw_slope);
A_y_slope_projected = calculate_y_slope(v11, A_slope);
delta_y_slope_projected = calculate_y_slope(v11, delta_slope);
q_y_slope_projected = calculate_y_slope(v11, q_slope);
lambda_y_slope_projected = calculate_y_slope(v11, lambda_slope);
tc_y_slope_projected = calculate_y_slope(v11, tc_slope);
Nz_y_slope_projected = calculate_y_slope(v11, Nz_slope);
Wdg_y_slope_projected = calculate_y_slope(v11, Wdg_slope);
Wp_y_slope_projected = calculate_y_slope(v11, Wp_slope);

% Calculate the norms of the projected y slopes for each variable
norms_projected_slopes = [
    norm(Sw_y_slope_projected),
    norm(Wfw_y_slope_projected),
    norm(A_y_slope_projected),
    norm(delta_y_slope_projected),
    norm(q_y_slope_projected),
    norm(lambda_y_slope_projected),
    norm(tc_y_slope_projected),
    norm(Nz_y_slope_projected),
    norm(Wdg_y_slope_projected),
    norm(Wp_y_slope_projected)
];

% Variable names
variable_names = {'Sw', 'Wfw', 'A', 'delta', 'q', 'lambda', 'tc', 'Nz', 'Wdg', 'Wp'};

% Plot the bar plot
figure;
bar(norms_projected_slopes);
xlabel('Variables');
ylabel('Norm of Projected y Slopes');
title('Norms of Projected y Slopes for Each Variable');
set(gca, 'XTick', 1:length(variable_names), 'XTickLabel', variable_names);
ylim([0, max(norms_projected_slopes) + 0.1 * max(norms_projected_slopes)]);  % Adjust y-axis limit for clarity
grid on;

%% norm-wise similarity
norm_data = som_normalize(sMap,'range');

for i = 1:size(norm_data.codebook,2)-1
    norm_sim(i) = norm(norm_data.codebook(:,i)-norm_data.codebook(:,end));
end

norm_sim_scaled = max(norm_sim)-(norm_sim);
v= norm_sim_scaled;
% Scale the vector

scaled_v = v- median(v);
scaled_v = scaled_v / max(abs(v));
% Bar plot using the scaled_v values and variable_names as labels
figure;
bar(scaled_v);
xlabel('Variables');
ylabel('Similarity Value');
title('Norm-wise similarity (base:median)');
set(gca, 'XTick', 1:length(variable_names), 'XTickLabel', variable_names);
% ylim([0, max(scaled_v) + 0.1*max(scaled_v)]);  % Adjusting y limits for better visualization
ylim([-1,1])
grid on;



final_cb = generateFinalCodebook(20, 40);

for i = 1:size(final_cb,2)
    base_finder(i) = norm(final_cb(:,i)-norm_data.codebook(:,end));
end
disp(base_finder)
base_scaled_finder = max(base_finder)-(base_finder);
base_value = 0.5*(min(base_scaled_finder)+max(base_scaled_finder));
disp(base_value)

scaled_v = v- median(base_scaled_finder);
scaled_v = scaled_v / max(abs(v));
% Bar plot using the scaled_v values and variable_names as labels
figure;
bar(scaled_v);
xlabel('Variables');
ylabel('Similarity Value');
title('Norm-wise similarity (base:calculated)');
set(gca, 'XTick', 1:length(variable_names), 'XTickLabel', variable_names);
% ylim([0, max(scaled_v) + 0.1*max(scaled_v)]);  % Adjusting y limits for better visualization
ylim([-1,1])
grid on;

%% Define functions

function similarity = matrix_cosine_similarity(A, B)
    C=A;
    A=B;
    B=C;
    % Reshape the matrices into vectors
    A_vector = A(:);
    B_vector = B(:);

    % Project A_vector onto B_vector
    proj_A_on_B = (dot(A_vector, B_vector) / dot(B_vector, B_vector)) * B_vector;

    % Compute cosine similarity between A_vector and proj_A_on_B
    similarity = dot(A_vector, proj_A_on_B) / (norm(A_vector) * norm(proj_A_on_B));
end


function slope = calculate_slope(codebook_vec)
    data_ = reshape(codebook_vec, [20, 20]);
    data_ = flipud(data_);
    [dZdx, dZdy] = gradient(data_);
    avg_dZdx = mean(dZdx(2:end-1, 2:end-1), 'all');
    avg_dZdy = mean(dZdy(2:end-1, 2:end-1), 'all');
    slope = [avg_dZdx, avg_dZdy];
end

function y_slope = calculate_y_slope(output_codebook_vec, x_slope)
    data_ = reshape(output_codebook_vec, [20, 20]);
    data_ = flipud(data_);
    [dZdx, dZdy] = gradient(data_);
    [dZdx, dZdy] = reflectAboutPerpendicular(dZdx, dZdy, x_slope);
    avg_dZdx = mean(dZdx(2:end-1, 2:end-1), 'all');
    avg_dZdy = mean(dZdy(2:end-1, 2:end-1), 'all');
    y_slope = [avg_dZdx, avg_dZdy];
    a = y_slope';
    b = x_slope';
    proj_b_a = (dot(a, b) / dot(b, b)) * b;
    y_slope = proj_b_a';
end

% Define selective reflection function
function [reflected_dx, reflected_dy] = reflectAboutPerpendicular(dx, dy, normal_vector)
    % Define the perpendicular direction
    n = [normal_vector(1); normal_vector(2)];
    n = n / norm(n); % Normalize
    
    % Compute reflection matrix
    R = eye(2) - 2 * (n * n');
    
    % Apply reflection matrix conditionally
    for i = 1:size(dx, 1)
        for j = 1:size(dy, 2)
            if dot([dx(i,j); dy(i,j)], n) < 0
                result = R * [dx(i,j); dy(i,j)];
                reflected_dx(i,j) = result(1);
                reflected_dy(i,j) = result(2);
            else
                reflected_dx(i,j) = dx(i,j);
                reflected_dy(i,j) = dy(i,j);
            end
        end
    end
end

function fx = wing_weight(Sw, Wfw, A, delta, q, lambda, tc, Nz, Wdg, Wp)
    % Wing weight function based on the provided formula
    fx = 0.0368 * Sw^0.758 * Wfw^0.0035 ...
        * (A/cos(delta)^2)^0.6 * q^0.006 * lambda^0.04 ...
        * (100 * tc / cos(delta))^-0.3 ...
        * (Nz * Wdg)^0.49 ...
        + Sw * Wp;
end


function final_codebook = generateFinalCodebook(n, num_directions)
    % INPUT:
    % n: size of the matrix (typically 20)
    % num_directions: number of equidistant directions in [0, 2*pi)

    final_codebook = zeros(n*n, num_directions);
    
    % Divide [0, 2pi) into equidistant angles
    angles = linspace(0, 2*pi, num_directions+1);
    angles = angles(1:end-1);  % Remove the last point as it's same as the first one

    for i = 1:num_directions
        % Calculate direction based on angle
        dx = cos(angles(i));
        dy = sin(angles(i));
        direction = [dx, dy];
        
        % Generate codebook vector for this direction
        codebook_vector = generateCodebook(n, direction);
        
        % Add the codebook vector to final_codebook
        final_codebook(:, i) = codebook_vector;
    end
end

% Helper function to generate codebook vector for a given direction
function codebook_vector = generateCodebook(n, direction)
    % INPUT:
    % n: size of the matrix (typically 20)
    % direction: 2-element vector specifying the slope in x and y direction

    % Calculate a, b, c based on direction
    a = direction(1);
    b = direction(2);
    c = -1;

    % Choose a reference height d at (x=1, y=1). For simplicity, let's choose it as 0.
    d = 0;

    % Compute the Z matrix
    Z = zeros(n, n);
    for x = 1:n
        for y = 1:n
            Z(y, x) = -a*x - b*y - d;   % Note the indices: Z(y, x)
        end
    end

    % Normalize Z values to [0, 1]
    Z = (Z - min(Z(:))) / (max(Z(:)) - min(Z(:)));

    % Convert the n x n Z matrix into a single codebook vector
    codebook_vector = reshape(Z, [], 1);
end