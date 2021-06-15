using Base: thread_notifiers
# Kalman filter exact initialisation 

using DataFrames
using Optim 
using LinearAlgebra


import XLSX



# ---------------------------------------------------------
# Function 1 - initialisation
# Function 1 - Kalman filter recursions
# Function 2 - Log Likelihood estimation
# Function 3 - Kalman smoother recursions
# ---------------------------------------------------------


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Mod
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  

mutable struct dlm  
    FF
    A
    H
    Q
    R
    k
    n
    cur_ξ_hat
    cur_Σ_hat
end


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Initialisation
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Function to initialse filter
#   1) Diffuse
#   2) Exact diffuse


# Sets up dlm, given system matrices
 # ξ_t = FFξ_(t-1) + Qw_t
 #  y_t = A'x_t + H'ξ_t + Rv_t 

 # TODO: Expand to inlcude x and y?
function dlm(A, FF, Q, R, H)
    k = size(H, 2)   # Number of y variables (columns of H as it is transposed)
    n = size(H, 1)   # Number of states (row of H as it is transposed
    ξ = n == 1 ? zero(eltype(FF)) : zeros(n)                    # IF n == 1 then 0, else zeros[n,1]
    Σ = n == 1 ? zero(eltype(FF)) : zeros(n, n)                 # IF n == 1 then 0, else zeros[n,n]   
    return dlm(A, FF, Q, R, H, k, n, ξ, Σ)
end


# User defined initialisation
function initialise!(m::dlm, ξ, Σ)
    m.cur_ξ_hat = ξ
    m.cur_Σ_hat = Σ
    Nothing
end

# Exact limit 

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Kalman filter
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ξ_t = Fξ_(t-1) + w_t
# y_t = A'x_t + H'ξ_t + v_t
# Assumes states have been initialised

function KalmanFilter(m::dlm,y::AbstractArray, x = "NA")

    FF, A, H, Q = m.FF, m.A, m.H, m.Q
      
    k, n, = m.k, m.n
    
    # *********************************
    # Initialisation of matrices
    # *********************************

    # Number of time periods
    T = size(y,1)

    ξ_f = zeros(T,n)
    Σ_f = zeros(n,n,T)
    
    ξ_f[1,n] = m.cur_ξ_hat
    Σ_f[n,n,1] = m.cur_Σ_hat

    ξ_p = zeros(T,n)
    Σ_p = zeros(n,n,T)

    if x == "NA"
        x = zeros(T,1)
    end    
    
    for t= 1:T

        # ******************************
        # Prediction step 
        # ******************************
            
        ξ_p[t,:] = FF*ξ_f[t,n]     
        
        Σ_p[:,:,t] = FF*Σ_f[n,n,t]*transpose(FF) + Q

        # Ensure Y vector is y[vars,time]
        if m.k > 1
            reshape(y[t], m.k, 1)
        end

        # Prediction error
        Hp = transpose(H)
        Ap = transpose(A)

        prediction_error = (y[t].-Ap*x[t].-Hp*Σ_p[n,n,t])

        # Prediction variance
        HΣHR = transpose(H)*Σ_p[n,n,t]*H .+ R

        # ******************************
        # Filtered step 
        # ******************************
        
        # Kalman Gain
        Gain = (Σ_p[n,n,t]*H)/(HΣHR)

        # Filtered mean
        ξ_f[t,:]  = ξ_p[t,n] .+ Gain*prediction_error

        # Filtered variance
        Σ_f[:,:,t] = Σ_p[n,n,t] .- Gain*H*Σ_p[n,n,t]

    end    
  
    return ξ_p,Σ_p,ξ_f,Σ_f

end




# Nile river data for testing
dat =  DataFrame(XLSX.readtable("C:/Users/aelde/OneDrive/Documents/GitHub/Tutorials/State space models/NileDat.xlsx","Sheet1")...)


# Sets up dlm, given system matrices
 # ξ_t = FFξ_(t-1) + Qw_t
 #  y_t = A'x_t + H'ξ_t + Rv_t 

 A = [0.0]
 H = [1.0]
 FF = [1.0]
 Q = [100.0]
 R = [1000.0]

mod1= dlm(FF,A,H,Q,R)
a =KalmanFilter(mod1,dat[:,1])
