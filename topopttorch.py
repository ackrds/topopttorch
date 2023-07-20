import numpy as np
import torch

def top3d(nelx, nely, nelz, volfrac, penal, rmin):
    # USER-DEFINED LOOP PARAMETERS
    maxloop = 200    # Maximum number of iterations
    tolx = 0.1       # Termination criterion
    displayflag = 0  # Display structure flag

    # USER-DEFINED MATERIAL PROPERTIES
    E0 = 1           # Young's modulus of solid material
    Emin = 1e-9      # Young's modulus of void-like material
    nu = 0.3         # Poisson's ratio

    # USER-DEFINED LOAD DOFs
    il = nelx // 2
    jl = nely
    kl = nelz // 2
    loadnid = torch.tensor(kl * (nelx + 1) * (nely + 1) + il * (nely + 1) + (nely + 1 - jl)) # Node IDs
    loaddof = 3 * loadnid - 1 # DOFs

    # USER-DEFINED SUPPORT FIXED DOFs
    iif = torch.tensor([0, 0, nelx, nelx])
    jf = torch.tensor([0, 0, 0, 0])
    kf = torch.tensor([0, nelz, 0, nelz])
    fixednid = kf * (nelx + 1) * (nely + 1) + iif * (nely + 1) + (nely + 1 - jf)  # Node IDs
    fixeddof = torch.cat([3 * fixednid, 3 * fixednid - 1, 3 * fixednid - 2])  # DOFs

    # PREPARE FINITE ELEMENT ANALYSIS
    nele = nelx * nely * nelz
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    
    
    size = torch.Size([ndof, 1])
    F = torch.sparse_coo_tensor(
    indices=torch.tensor([[loaddof, j] for j in range(size[1])]).t(),
    values=torch.full((size[1],), -1, dtype=torch.float32),
    size=size,
)

        
    U = torch.zeros(ndof, 1)
    freedofs = torch.tensor(list(set(range(1, ndof + 1)) - set(fixeddof.tolist())))
    KE = lk_H8(nu)
    nodegrd = torch.arange((nely + 1) * (nelx + 1)).reshape(nely + 1, nelx + 1)
    nodeids = nodegrd[:-1, :-1].reshape(-1, 1)
    nodeidz = torch.arange(0, (nely + 1) * (nelx + 1) * (nelz - 1) + 1, (nely + 1) * (nelx + 1))
    nodeids = nodeids.repeat(1, nodeidz.shape[0]) + nodeidz.repeat(nodeids.shape[0], 1)
    edofVec = 3 * nodeids + 1
    
    offsets = torch.tensor([0, 1, 2, 3 * nely + torch.tensor([3, 4, 5, 0, 1, 2]), -3, -2, -1, 3 * (nely + 1) * (nelx + 1) + torch.tensor([0, 1, 2, 3 * nely + torch.tensor([3, 4, 5, 0, 1, 2]), -3, -2, -1])])
    edofMat = edofVec.unsqueeze(1) + offsets.unsqueeze(0)  # Perform element-wise addition with broadcasting
    
    # Reshape iK and jK
    edofMat_flat = edofMat.flatten()
    iK = torch.reshape(torch.kron(edofMat_flat, torch.ones(24)), (24 * 24 * nele,))
    jK = torch.reshape(torch.kron(edofMat_flat, torch.ones(1, 24)), (24 * 24 * nele,))
    
    # PREPARE FILTER
    rmin = torch.ceil(rmin)  # Assuming rmin is already defined
    iH = torch.ones(nele * (2 * (rmin.long() - 1) + 1) ** 2)
    jH = torch.ones_like(iH)
    sH = torch.zeros_like(iH)
    k = 0
    

    nele = nelx * nely * nelz
    iH = []
    jH = []
    sH = []
    k = 0
    
    for k1 in range(1, nelz + 1):
        for i1 in range(1, nelx + 1):
            for j1 in range(1, nely + 1):
                e1 = (k1 - 1) * nelx * nely + (i1 - 1) * nely + j1
                k1_min = max(k1 - (torch.ceil(rmin).long() - 1), 1)
                k1_max = min(k1 + (torch.ceil(rmin).long() - 1), nelz)
                i1_min = max(i1 - (torch.ceil(rmin).long() - 1), 1)
                i1_max = min(i1 + (torch.ceil(rmin).long() - 1), nelx)
                j1_min = max(j1 - (torch.ceil(rmin).long() - 1), 1)
                j1_max = min(j1 + (torch.ceil(rmin).long() - 1), nely)
    
                for k2 in range(k1_min, k1_max + 1):
                    for i2 in range(i1_min, i1_max + 1):
                        for j2 in range(j1_min, j1_max + 1):
                            e2 = (k2 - 1) * nelx * nely + (i2 - 1) * nely + j2
                            k += 1
                            iH.append(e1)
                            jH.append(e2)
                            sH.append(max(0, rmin - torch.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2 + (k1 - k2) ** 2)))
    
    iH = torch.tensor(iH)
    jH = torch.tensor(jH)
    sH = torch.tensor(sH)
    
    H = torch.sparse_coo_tensor(torch.stack([iH, jH]), sH, (nele, nele))
    Hs = torch.sum(H, dim=1)

    x = torch.full((nely, nelx, nelz), volfrac)
    xPhys = x
    loop = 0
    change = 1

    while change > tolx and loop < maxloop:
        loop += 1
        # FE-ANALYSIS
        sK = torch.reshape(KE.flatten() * (Emin + xPhys.flatten() ** penal * (E0 - Emin)), (24 * 24 * nele, 1))
        K = torch.sparse_coo_tensor(torch.stack([iK, jK]), sK, (24 * nele, 24 * nele)).to_dense()
        K = (K + K.t()) / 2
        U = torch.zeros_like(F)
        U[freedofs, :] = torch.solve(F[freedofs, :], K[freedofs, :][:, freedofs]).solution
        # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
        U_edofMat = torch.index_select(U, 0, edofMat.flatten())
        ce = torch.reshape(torch.sum((U_edofMat * KE) * U_edofMat, dim=1), (nely, nelx, nelz))
        c = torch.sum((Emin + xPhys ** penal * (E0 - Emin)) * ce)
        dc = -penal * (E0 - Emin) * xPhys ** (penal - 1) * ce
        dv = torch.ones_like(xPhys)
        # FILTERING AND MODIFICATION OF SENSITIVITIES
        dc = torch.sparse_coo_tensor(torch.stack([iH, jH]), dc.flatten() / Hs, (nele, 1)).to_dense()
        dv = torch.sparse_coo_tensor(torch.stack([iH, jH]), dv.flatten() / Hs, (nele, 1)).to_dense()
        dc = torch.matmul(H, dc)
        dv = torch.matmul(H, dv)

        l1 = 0
        l2 = 1e9
        move = 0.2
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew = torch.clamp(x - move, 0, 1)
            xnew = torch.minimum(xnew + move, x * torch.sqrt(-dc / (dv * lmid)))
            xnew = torch.maximum(xnew, 0)
            xPhys = torch.matmul(H, xnew.flatten()) / Hs
            if torch.sum(xPhys) > volfrac * nele:
                l1 = lmid
            else:
                l2 = lmid
        change = torch.max(torch.abs(xnew - x))
        x = xnew
        # PRINT RESULTS
        print(f' It.:{loop:5d} Obj.:{c:11.4f} Vol.:{torch.mean(xPhys):7.3f} ch.:{change:7.3f}')
        # PLOT DENSITIES
        if displayflag:
            # Plotting code for displaying 3D densities
            pass
    




def lk_H8(nu):
    A = np.array([[32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
                  [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12]], dtype=np.float32)

    # Calculate k using NumPy operations
    k = (1 / 144) * np.matmul(A.T, np.array([1, nu], dtype=np.float32).reshape(-1, 1))

    # Split k into submatrices
    K1 = np.array([[k[0], k[1], k[1], k[2], k[4], k[4]],
                   [k[1], k[0], k[1], k[3], k[5], k[6]],
                   [k[1], k[1], k[0], k[3], k[6], k[5]],
                   [k[2], k[3], k[3], k[0], k[7], k[7]],
                   [k[4], k[5], k[6], k[7], k[0], k[1]],
                   [k[4], k[6], k[5], k[7], k[1], k[0]]]).squeeze()

    K2 = np.array([[k[8],  k[7],  k[11], k[5],  k[3],  k[6]],
                   [k[7],  k[8],  k[11], k[4],  k[2],  k[5]],
                   [k[9],  k[9],  k[12], k[6],  k[4],  k[5]],
                   [k[5],  k[4],  k[10], k[8],  k[1],  k[9]],
                   [k[3],  k[2],  k[4],  k[1],  k[8],  k[11]],
                   [k[10], k[4],  k[6],  k[12], k[10], k[12]]]).squeeze()

    K3 = np.array([[k[6],  k[7],  k[4],  k[9],  k[12], k[8]],
                   [k[7],  k[6],  k[4],  k[10], k[13], k[10]],
                   [k[5],  k[5],  k[3],  k[8],  k[12], k[9]],
                   [k[9],  k[10], k[2],  k[6],  k[11], k[5]],
                   [k[12], k[13], k[10], k[11], k[6],  k[4]],
                   [k[2],  k[12], k[9],  k[4],  k[5],  k[3]]]).squeeze()

    K4 = np.array([[k[13], k[10], k[10], k[12], k[9],  k[9]],
                   [k[10], k[13], k[10], k[11], k[8],  k[7]],
                   [k[10], k[10], k[13], k[11], k[7],  k[8]],
                   [k[12], k[11], k[11], k[13], k[6],  k[6]],
                   [k[9],  k[8],  k[7],  k[6],  k[13], k[10]],
                   [k[9],  k[7],  k[8],  k[6],  k[10], k[13]]]).squeeze()

    K5 = np.array([[k[0], k[1],  k[7],  k[2], k[4],  k[3]],
                   [k[1], k[0],  k[7],  k[3], k[5],  k[10]],
                   [k[7], k[7],  k[0],  k[4], k[10], k[5]],
                   [k[2], k[3],  k[4],  k[0], k[7],  k[1]],
                   [k[4], k[5],  k[10], k[7], k[0],  k[7]],
                   [k[3], k[10], k[5],  k[1], k[7],  k[0]]]).squeeze()

    K6 = np.array([[k[13], k[10], k[6],  k[12], k[9],  k[11]],
                   [k[10], k[13], k[6],  k[11], k[8],  k[2]],
                   [k[6],  k[6],  k[13], k[10], k[2],  k[9]],
                   [k[12], k[11], k[10], k[13], k[7],  k[7]],
                   [k[9],  k[8],  k[2],  k[7],  k[13], k[7]],
                   [k[11], k[2],  k[9],  k[10], k[7],  k[13]]]).squeeze()
    
    print(K1.shape,  K2.shape, K3.shape, K4.shape, K5.shape, K6.shape)
    # Assemble the global stiffness matrix KE
    KE = (1 / ((nu + 1) * (1 - 2 * nu))) * np.block([[K1, K2, K3, K4],
                                                     [K2.T, K5, K6, K3.T],
                                                     [K3.T, K6, K5.T, K2.T],
                                                     [K4, K3, K2, K1]])

    return KE

# Example usage:
    return KE

if __name__ == "__main__":
    top3d(20,20,20,0.4, 3.0, 1.5)
