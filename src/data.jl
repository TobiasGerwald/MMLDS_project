using OrdinaryDiffEq
#for loading El Nino Data
using NetCDF

### Create training Data ###

function lorenz(u, p, t)
    x, y, z = u
    σ, ρ, β = p
    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = x * y - β * z
return [dx, dy, dz]
end

function create_data(x0 = [1.,1.,1.],dt = 0.01)

    p_lorenz = [10, 28, 8/3] 
    tstart = 0.
    tend = 100.
    tspan = (tstart, tend)
    saveat = tstart:dt:tend

    prob = ODEProblem(lorenz, x0, tspan, p_lorenz) 
    sol = solve(prob, Tsit5(), saveat=saveat)

    return sol

end

#Loading and changing El Nino Data
path_to_data = "sst.mon.mean.nc"

x = ncread(filename, "sst") #sst = Sea Surface Temperature
x_reduced = x[121:170, 86:95, :] #only concerned with the important region for the El Nino happening

function sum_elements_without_empty_values(y)
    h,w = size(y)
    counter = 0
    summation = 0
    for i in 1:h
        for j in 1:w
            if y[i,j] != 1.0f20
                summation = summation + y[i,j]
                counter = counter + 1
            end
        end
    end
    return summation/counter #using the mean here
end

function sum_elements_without_empty_values_vec(y) #EMPTY VS ZERO
    bool_matrix = y .< 1.0f20 
    nonzero_elements = sum(bool_matrix)
    y_adjusted = y .* bool_matrix
    summation = sum(y_adjusted)
    return summation/nonzero_elements
end

function compress_data_matrix(x_data, kernel_size, vectorized = true)
    h,w,d = size(x_data)
    n_h = trunc(Int, h/kernel_size)
    n_w = trunc(Int, w/kernel_size)
    A = ones(n_h, n_w,d)
    for dim in 1:d
        for i in 1:n_h
            filter_region_h = (1 + (i-1)*kernel_size): (i*kernel_size)
            for j in 1:n_w
                filter_region_w = (1 + (j-1)*kernel_size): (j*kernel_size)
                y = x_reduced[filter_region_h, filter_region_w, dim]
                if vectorized
                    A[i,j,dim] = sum_elements_without_empty_values_vec(y)
                else
                    A[i,j,dim] = sum_elements_without_empty_values(y)
                end
            end
        end
    end
    return A
end
