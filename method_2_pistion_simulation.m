clc
clear all
close all

%% Problem definition
n = 500;

% Bounds for each variable based on the provided values
bounds = {
    'M', 30, 60;
    'S', 0.005, 0.020;
    'Vo', 0.002, 0.010;
    'k', 1000, 5000;
    'Po', 90000, 110000;
    'Ta', 290, 296;
    'To', 340, 360;
};

% Generate data for each variable
for i = 1:size(bounds, 1)
    eval([bounds{i, 1} ' = ' num2str(bounds{i, 2}) ' + (' num2str(bounds{i, 3}-bounds{i, 2}) ')*rand(n, 1);']);
end

% Calculate A and V
A = Po .* S + 19.62 .* M - (k .* Vo ./ S);
V = (S ./ (2.*k)) .* (sqrt(A.^2 + 4.*k.*(Po.*Vo./To).*Ta) - A);

% Calculate the piston simulation output
C = 2 * pi * sqrt(M ./ (k + S.^2 .* (Po.*Vo./To) .* (Ta./V.^2)));

% Concatenate into a data matrix
data = [M k S Po Vo To Ta C];

% Assuming your piston simulation data is in a matrix 'piston_data'

% Perform PCA on your piston data (excluding the 'C' column)
[coeff_piston, score_piston, latent_piston, tsquared_piston, explained_piston] = pca(data(:,1:7));

% Display explained variance for piston simulation data
disp('Explained Variance Ratios for Piston Simulation:');
disp(explained_piston(1:2));

% Display factor loadings for the first two principal components for piston simulation data
disp('Factor Loadings (First two components) for Piston Simulation:');
disp(coeff_piston(:,1:2));


%% Dataset
comp_names = {'M', 'k', 'S', 'Po', 'Vo', 'To', 'Ta', 'C'};
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
figure(1) 
som_show(sMap, 'umat','all','comp', 'all');

%% iSOM Grid in function space (using just the first three components for visualization)
figure(2)
som_grid(sMap,'coord',sMap.codebook(:,[1 2 8]),'label',sMap.labels,'labelcolor','c','labelsize',10, 'marker','o','MarkerColor','k',...
    'MarkerSize',7,'linecolor', 'k');
hold on, scatter3(data(:,1),data(:,2),data(:,8),20,'ro','filled');
xlabel('M')
ylabel('k')
zlabel('C')


%% Cosine similarity
vM = sMap.codebook(:,1); 
vk = sMap.codebook(:,2); 
vS = sMap.codebook(:,3); 
vPo = sMap.codebook(:,4); 
vVo = sMap.codebook(:,5); 
vTa = sMap.codebook(:,6); 
vTo = sMap.codebook(:,7); 
vC = sMap.codebook(:,8); % C (output)

% Calculate cosine similarity between each variable vector and C
theta_M = 180 * acos((vM'*vC) / (norm(vM) * norm(vC))) / pi;
theta_k = 180 * acos((vk'*vC) / (norm(vk) * norm(vC))) / pi;
theta_S = 180 * acos((vS'*vC) / (norm(vS) * norm(vC))) / pi;
theta_Po = 180 * acos((vPo'*vC) / (norm(vPo) * norm(vC))) / pi;
theta_Vo = 180 * acos((vVo'*vC) / (norm(vVo) * norm(vC))) / pi;
theta_Ta = 180 * acos((vTa'*vC) / (norm(vTa) * norm(vC))) / pi;
theta_To = 180 * acos((vTo'*vC) / (norm(vTo) * norm(vC))) / pi;

%% Set the normalization value (you can change this to 90 or 180 as needed)
normalization_angle = 90;  % or 180
% normalization_angle = max([theta_M, theta_k, theta_S, theta_Po, theta_Vo, theta_Ta, theta_To]);
%% Calculate Normalized Importance based on Cosine similarity
% Normalize and invert angles to get importance values
importance_M = 1 - (theta_M/normalization_angle);
importance_k = 1 - (theta_k/normalization_angle);
importance_S = 1 - (theta_S/normalization_angle);
importance_Po = 1 - (theta_Po/normalization_angle);
importance_Vo = 1 - (theta_Vo/normalization_angle);
importance_Ta = 1 - (theta_Ta/normalization_angle);
importance_To = 1 - (theta_To/normalization_angle);

% Compile importance scores
importance_scores = [importance_M, importance_k, importance_S, importance_Po, importance_Vo, importance_Ta, importance_To];
variable_names = {'M', 'k', 'S', 'Po', 'Vo', 'Ta', 'To'};

%% Display Importance Scores as a Bar Plot
% figure;
% bar(importance_scores);
% xlabel('Variables');
% ylabel('Normalized Importance');
% title(['Normalized Importance of Variables based on Cosine Similarity (Normalized with ' num2str(normalization_angle) ' degrees)']);
% set(gca, 'XTick', 1:length(variable_names), 'XTickLabel', variable_names);
% ylim([0 1]);


disp('Explained Variance Ratios for Piston Simulation:');
disp(explained_piston(1:2));

% Display factor loadings for the first two principal components for piston simulation data
disp('Factor Loadings (First two components) for Piston Simulation:');
disp(coeff_piston(:,1:2));

% Compute slopes for each variable in piston simulation
M_slope = calculate_slope(vM);
k_slope = calculate_slope(vk);
S_slope = calculate_slope(vS);
Po_slope = calculate_slope(vPo);
Vo_slope = calculate_slope(vVo);
Ta_slope = calculate_slope(vTa);
To_slope = calculate_slope(vTo);

% Compute the projected y slopes for each variable in piston simulation
M_y_slope_projected = calculate_y_slope(vC, M_slope);
k_y_slope_projected = calculate_y_slope(vC, k_slope);
S_y_slope_projected = calculate_y_slope(vC, S_slope);
Po_y_slope_projected = calculate_y_slope(vC, Po_slope);
Vo_y_slope_projected = calculate_y_slope(vC, Vo_slope);
Ta_y_slope_projected = calculate_y_slope(vC, Ta_slope);
To_y_slope_projected = calculate_y_slope(vC, To_slope);

% Calculate the norms of the projected y slopes for each variable in piston simulation
norms_projected_slopes_piston = [
    norm(M_y_slope_projected),
    norm(k_y_slope_projected),
    norm(S_y_slope_projected),
    norm(Po_y_slope_projected),
    norm(Vo_y_slope_projected),
    norm(Ta_y_slope_projected),
    norm(To_y_slope_projected)
];

% Variable names for piston simulation
variable_names_piston = {'M', 'k', 'S', 'Po', 'Vo', 'Ta', 'To'};

% Bar plot for piston simulation data
figure;
bar(norms_projected_slopes_piston);
xlabel('Variables (Piston Simulation)');
ylabel('Norm of Projected y Slopes (Piston Simulation)');
title('Norms of Projected y Slopes for Each Variable in Piston Simulation');
set(gca, 'XTick', 1:length(variable_names_piston), 'XTickLabel', variable_names_piston);
ylim([0, max(norms_projected_slopes_piston) + 0.1 * max(norms_projected_slopes_piston)]);
grid on;

%% Norm-wise similarity for piston simulation
norm_data_piston = som_normalize(sMap,'range');

for i = 1:size(norm_data_piston.codebook,2)-1
    norm_sim_piston(i) = norm(norm_data_piston.codebook(:,i)-norm_data_piston.codebook(:,end));
end


% norm_sim_scaled_piston = max(norm_sim_piston)-(norm_sim_piston);
norm_sim_scaled_piston = max(norm_sim_piston)-(norm_sim_piston);
disp(norm_sim_scaled_piston)
v_piston = norm_sim_scaled_piston;
disp("median")
disp(median(v_piston))

% Scale the vector for piston simulation
scaled_v_piston = v_piston - median(v_piston)
% out = norm_data_piston.codebook(:,end);

% mid = 0.5*norm(out-(1-out));
% scaled_v_piston = v_piston -0.5*norm(out-(1-out));
scaled_v_piston = scaled_v_piston / max(abs(v_piston));
% Bar plot using the scaled_v_piston values and variable_names_piston as labels
figure;
bar(scaled_v_piston);
xlabel('Variables (Piston Simulation)');
ylabel('Similarity Value (Piston Simulation)');
title('Norm-wise similarity (base:median)');
set(gca, 'XTick', 1:length(variable_names_piston), 'XTickLabel', variable_names_piston);
% ylim([min(scaled_v_piston), max(scaled_v_piston) + 0.1*max(scaled_v_piston)]);
ylim([-1,1]);
grid on;


final_cb = generateFinalCodebook(20, 10);

for i = 1:size(final_cb,2)
    base_finder(i) = norm(final_cb(:,i)-norm_data_piston.codebook(:,end));
end

base_scaled_finder = max(base_finder)-(base_finder);
disp(base_scaled_finder)
base_value = 0.5*(min(base_scaled_finder)+max(base_scaled_finder));
disp(base_value)


% Scale the vector for piston simulation
scaled_v_piston = v_piston - base_value
% out = norm_data_piston.codebook(:,end);

% mid = 0.5*norm(out-(1-out));
% scaled_v_piston = v_piston -0.5*norm(out-(1-out));
scaled_v_piston = scaled_v_piston / max(abs(v_piston));
% Bar plot using the scaled_v_piston values and variable_names_piston as labels
figure;
bar(scaled_v_piston);
xlabel('Variables (Piston Simulation)');
ylabel('Similarity Value (Piston Simulation)');
title('Norm-wise similarity (base:calculated)');
set(gca, 'XTick', 1:length(variable_names_piston), 'XTickLabel', variable_names_piston);
% ylim([min(scaled_v_piston), max(scaled_v_piston) + 0.1*max(scaled_v_piston)]);
ylim([-1,1]);
grid on;


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