using Base: thread_notifiers
# Kalman filter exact initialisation 

using DataFrames
using Optim 
using LinearAlgebra

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
    n = size(H, 1)   # Number of states (ros of H as it is transposed
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

function KalmanFilter!(m::dlm,y::AbstractArray,x::AbstractArray)

 
    FF, A, H, X, Q = m.FF, m.A, m.H, m.Q
    ξ_p1 , Σ_pi = m.cur_ξ_hat, m.cur_Σ_hat 

    # Number of time periods
    bigT = size(y,1)
         
    # ******************************
    # Prediction step 
    # ******************************
           
    ξ_p1 = FF*ξ_f1     
    Σ_p1 = FF*Σ_f1*transpose(FF) + Q

    # Ensure Y vector is y[vars,time]
    if m.k > 1
        reshape(y[t], m.k, 1)
    end

    # Prediction error
    prediction.error = (y[t]-.transpose(A)*x-.transpose(H)*ξ_p1)

    # Prediction variance
    HΣHR = transpose(H)*Σ_p1*H + R

    # ******************************
    # Filtered step 
    # ******************************
    


    # Kalman Gain
    Gain = (Σ_p1*H)/(HΣHR)

    # Filtered mean
    m.cur_ξ_hat  = ξ_p1 + Gain*prediction.error

    # Filtered variance
    m.cur_Σ_hat = Σ_p1 - Gain*H*Σ_p1

  
end

