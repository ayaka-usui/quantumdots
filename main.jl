
using Arpack, SparseArrays, LinearAlgebra
# using ExpmV
using Plots

function createH!(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,matH::SparseMatrixCSC{Float64})

    # matH = sparse(Float64,K*2+1,K*2+1)
    Depsilon = W/(K-1)

    matH[1,1] = 0.0 # epsilon for the system, probably 0

    for kk = 1:K
        matH[1+kk,1+kk] = (kk-1)*Depsilon - W/2 # epsilon for the bath L
        matH[1+kk,1] = sqrt(GammaL*Depsilon/(2*pi)) # tunnel with the bath L
        matH[1+K+kk,1] = sqrt(GammaR*Depsilon/(2*pi)) # tunnel with the bath R
    end
    matH[K+2:end,K+2:end] = matH[2:K+1,2:K+1] # epsilon for the bath R

    matH .= matH + matH' - spdiagm(diag(matH))

end

function calculatequantities2(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH!(K,W,betaL,betaR,GammaL,GammaR,matH)

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    # Depsilon = W/(K-1)

    Ct = zeros(ComplexF64,K*2+1,K*2+1)
    dCt = zeros(ComplexF64,K*2+1)
    Ct_E = zeros(ComplexF64,K*2,K*2)
    val_Ct_E = zeros(ComplexF64,K*2)
    val_Ct = zeros(ComplexF64,K*2+1)
    dCt1 = zeros(ComplexF64,K*2+1)

    epsilonLR = diag(matH)
    tLRk = matH[1,1:end]

    vNE_sys = zeros(ComplexF64,Nt)
    vNE_E = zeros(ComplexF64,Nt)
    vNE_alphak = zeros(ComplexF64,Nt)
    I_SE = zeros(ComplexF64,Nt)
    betaQL = zeros(ComplexF64,Nt)
    betaQR = zeros(ComplexF64,Nt)
    sigma = zeros(ComplexF64,Nt)
    Drel = zeros(ComplexF64,Nt)
    I_env = zeros(ComplexF64,Nt)

    E_sys = zeros(ComplexF64,Nt)
    E_L = zeros(ComplexF64,Nt)
    E_R = zeros(ComplexF64,Nt)
    F_L = zeros(ComplexF64,Nt)
    F_R = zeros(ComplexF64,Nt)
    E_tot = zeros(ComplexF64,Nt)
    vNE = zeros(ComplexF64,Nt)
    N_sys = zeros(ComplexF64,Nt)
    N_L = zeros(ComplexF64,Nt)
    N_R = zeros(ComplexF64,Nt)

    for tt = 1:Nt

        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH
        # Ct = Hermitian(Ct)

        # energy
        dCt .= diag(Ct) #diag(Ct - C0)
        E_sys[tt] = dCt[1]*epsilonLR[1]
        E_L[tt] = sum(dCt[2:K+1].*epsilonLR[2:K+1])
        E_R[tt] = sum(dCt[K+2:2*K+1].*epsilonLR[K+2:2*K+1])
        # F_L[tt] = sum(dCt[2:K+1].*(epsilonLR[2:K+1] .- muL))
        # F_R[tt] = sum(dCt[K+2:2*K+1].*(epsilonLR[K+2:2*K+1] .- muR))
        # E_tot[tt] = sum(dCt[1:end].*epsilonLR[1:end])

        dCt1 .= Ct[1,1:end] # Ct[1,1:end] - C0[1,1:end]
        E_tot[tt] = E_L[tt] + E_R[tt] + sum(dCt1[2:end].*tLRk[2:end])*2

        # particle numbers
        N_sys[tt] = dCt[1]
        N_L[tt] = sum(dCt[2:K+1])
        N_R[tt] = sum(dCt[K+2:2*K+1])

        # vNE for total
        val_Ct .= eigvals(Ct)
        vNE[tt] = - sum(val_Ct.*log.(val_Ct)) - sum((1.0 .- val_Ct).*log.(1.0 .- val_Ct))

    end

    return time, E_sys, E_L, E_R, N_sys, N_L, N_R, E_tot

end

function calculatequantities(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH!(K,W,betaL,betaR,GammaL,GammaR,matH)

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    # Depsilon = W/(K-1)

    Ct = zeros(ComplexF64,K*2+1,K*2+1)
    dCt = zeros(ComplexF64,K*2+1)
    Ct_E = zeros(ComplexF64,K*2,K*2)
    val_Ct_E = zeros(ComplexF64,K*2)
    epsilonLR = zeros(ComplexF64,K*2+1)

    vNE_sys = zeros(ComplexF64,Nt)
    vNE_E = zeros(ComplexF64,Nt)
    vNE_alphak = zeros(ComplexF64,Nt)
    I_SE = zeros(ComplexF64,Nt)
    betaQL = zeros(ComplexF64,Nt)
    betaQR = zeros(ComplexF64,Nt)
    sigma = zeros(ComplexF64,Nt)
    Drel = zeros(ComplexF64,Nt)
    I_env = zeros(ComplexF64,Nt)

    for tt = 1:Nt

        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH
        # Ct = Hermitian(Ct)

        # vNE
        vNE_sys[tt] = -Ct[1,1]*log(Ct[1,1]) - (1-Ct[1,1])*log(1-Ct[1,1])
        Ct_E .= Ct[2:end,2:end]
        val_Ct_E .= eigvals(Ct_E)
        vNE_E[tt] = - sum(val_Ct_E.*log.(val_Ct_E)) - sum((1.0 .- val_Ct_E).*log.(1.0 .- val_Ct_E))

        # I_SE
        I_SE[tt] = vNE_sys[tt] - vNE_sys[1] + vNE_E[tt] - vNE_E[1]

        # heat
        dCt .= diag(Ct - C0)
        epsilonLR .= diag(matH)
        betaQL[tt] = -sum(dCt[2:K+1].*(epsilonLR[2:K+1] .- muL))*betaL
        betaQR[tt] = -sum(dCt[K+2:2*K+1].*(epsilonLR[K+2:2*K+1] .- muR))*betaR

        # entropy production
        sigma[tt] = vNE_sys[tt] - vNE_sys[1] - betaQL[tt] - betaQR[tt]

        # relative entropy
        Drel[tt] = - betaQL[tt] - betaQR[tt] - (vNE_E[tt] - vNE_E[1])

        # mutual information describing the intraenvironment correlations
        diag_Ct_E = diag(Ct_E)
        vNE_alphak[tt] = - sum(diag_Ct_E.*log.(diag_Ct_E)) - sum((1.0 .- diag_Ct_E).*log.(1.0 .- diag_Ct_E))
        I_env[tt] = vNE_alphak[tt] - vNE_E[tt]

    end

    return time, vNE_E.-vNE_E[1], I_SE, betaQL, betaQR, sigma, Drel, I_env

end