import numpy as np
import os, sys
import gmsh


# ========================================================= #
# ===  generate pole parts of the magnet                === #
# ========================================================= #

def generate__poleLayer( lc=0.0, side="+", height=0.8 ):

    # ------------------------------------------------- #
    # --- [1] preparation                           --- #
    # ------------------------------------------------- #
    #  -- [1-1] constants                           --  #
    eps      = 1.e-8
    x_,y_,z_ = 0, 1, 2
    origin   = [ 0.0, 0.0, 0.0 ]
    ptDim, lineDim, surfDim  = 0, 1, 2
    if ( side == "+" ):
        th1,th2 = -0.0, 90.0
    if ( side == "-" ):
        th1,th2 = 90.0, 270.0

    #  -- [1-2] load parameters                     --  #
    import nkUtilities.load__constants as lcn
    cnsFile  = "dat/parameter.conf"
    const    = lcn.load__constants( inpFile=cnsFile )
    radius   = const["radius"]

    #  -- [1-3] load point Data                     --  #
    inpFile        = "dat/onmesh.dat"
    import nkUtilities.load__pointFile as lpf
    pointData           = lpf.load__pointFile( inpFile=inpFile, returnType="point" )
    
    # ------------------------------------------------- #
    # --- [2] define surface for circumference cut  --- #
    # ------------------------------------------------- #
    #  -- [2-1] collect on Arc points               --  #
    radii          = np.sqrt( pointData[:,x_]**2 + pointData[:,y_]**2 )
    index          = np.where( ( radius-eps < radii ) & \
                               ( radius+eps > radii ) )
    circumf_points = pointData[index]
    index          = np.argsort( circumf_points[:,y_] )
    circumf_points = circumf_points[index][:]
    #  -- [2-2] point to be plot                    --  #
    circumf_ext1   = 0.7 * np.copy( circumf_points )
    circumf_ext2   = 1.3 * np.copy( circumf_points )
    nCircumf       = circumf_points.shape[0]
    #  -- [2-3] add points to the model             --  #
    pnums1, pnums2 = [], []
    for pt in circumf_ext1:
        pnum  = gmsh.model.occ.addPoint( pt[x_], pt[y_], pt[z_], meshSize=lc )
        pnums1.append( pnum )
    for pt in circumf_ext2:
        pnum  = gmsh.model.occ.addPoint( pt[x_], pt[y_], pt[z_], meshSize=lc )
        pnums2.append( pnum )
    #  -- [2-4] define quad                         --  #
    tools = []
    for ik in range( nCircumf-1 ):
        l1 = gmsh.model.occ.addLine( pnums1[ik  ], pnums1[ik+1] )
        l2 = gmsh.model.occ.addLine( pnums1[ik+1], pnums2[ik+1] )
        l3 = gmsh.model.occ.addLine( pnums2[ik+1], pnums2[ik  ] )
        l4 = gmsh.model.occ.addLine( pnums2[ik  ], pnums1[ik  ] )
        cl = gmsh.model.occ.addCurveLoop( [l1,l2,l3,l4] )
        sf = gmsh.model.occ.addPlaneSurface( [cl] )
        tools.append( sf )

    # ------------------------------------------------- #
    # --- [3] define circumference surface          --- #
    # ------------------------------------------------- #
    #  -- [3-1] side "+" case                       --  #
    if ( side == "+" ):
        pth1         = -90.0 / 180.0 * np.pi
        pth2         = +90.0 / 180.0 * np.pi
        arc1         = gmsh.model.occ.addCircle( origin[x_], origin[y_], origin[z_], \
                                                 radius, angle1=pth1, angle2=pth2 )
        dx,dy,dz     = 0.0, 0.0, height
        ret          = gmsh.model.occ.extrude( [(lineDim,arc1)], dx,dy,dz )
        circumf_surf = ret[1][1]
    #  -- [3-2] boolean cut of the surface          --  #
    gmsh.model.occ.synchronize()
    target = [(surfDim,circumf_surf)]
    tools  = [(surfDim,int( tool ) ) for tool in tools ]
    ret    = gmsh.model.occ.cut( target, tools, removeObject=False, removeTool=False )
    gmsh.model.occ.remove( tools, recursive=True )
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    # ------------------------------------------------- #
    # --- [4] generate floor & ceiling sector       --- #
    # ------------------------------------------------- #
    import nkGmshRoutines.generate__sector180 as sec
    floor   = sec.generate__sector180( r1=0.0, r2=radius, zoffset=   0.0, defineSurf=True )
    ceiling = sec.generate__sector180( r1=0.0, r2=radius, zoffset=height, defineSurf=True )

    # ------------------------------------------------- #
    # --- [5] define diameter division              --- #
    # ------------------------------------------------- #
    #  -- [5-1] collect on Diameter points          --  #
    index            = np.where( ( pointData[:,x_] >= - eps ) & \
                                 ( pointData[:,x_] <= + eps ) )
    onDia_points     = pointData[index]
    index            = np.argsort( onDia_points[:,y_] )
    onDia_points     = onDia_points[index][:]
    #  -- [5-2] point to be plot                     --  #
    square_width     = 0.1 * radius
    onDia_ext1       = np.copy( onDia_points )
    onDia_ext2       = np.copy( onDia_points )
    onDia_ext1[:,x_] = - square_width
    onDia_ext2[:,x_] = + square_width
    nDiameter        = onDia_points.shape[0]
    #  -- [5-3] add points to the model              --  #
    pnums1, pnums2 = [], []
    for pt in onDia_ext1:
        pnum  = gmsh.model.occ.addPoint( pt[x_], pt[y_], pt[z_], meshSize=lc )
        pnums1.append( pnum )
    for pt in onDia_ext2:
        pnum  = gmsh.model.occ.addPoint( pt[x_], pt[y_], pt[z_], meshSize=lc )
        pnums2.append( pnum )
    #  -- [5-4] define quad                          --  #
    tools = []
    for ik in range( nDiameter-1 ):
        l1 = gmsh.model.occ.addLine( pnums1[ik  ], pnums1[ik+1] )
        l2 = gmsh.model.occ.addLine( pnums1[ik+1], pnums2[ik+1] )
        l3 = gmsh.model.occ.addLine( pnums2[ik+1], pnums2[ik  ] )
        l4 = gmsh.model.occ.addLine( pnums2[ik  ], pnums1[ik  ] )
        cl = gmsh.model.occ.addCurveLoop( [l1,l2,l3,l4] )
        sf = gmsh.model.occ.addPlaneSurface( [cl] )
        tools.append( sf )
        
    # ------------------------------------------------- #
    # --- [6] define diameter surface               --- #
    # ------------------------------------------------- #
    #  -- [6-1] define quad surface                 --  #
    import nkGmshRoutines.generate__quadShape as gqs
    x1,x2,x3,x4  = [ [ 0.0, - radius,    0.0 ],
                     [ 0.0, + radius,    0.0 ],
                     [ 0.0, + radius, height ],
                     [ 0.0, - radius, height ] ]
    ret          = gqs.generate__quadShape( x1=x1, x2=x2, x3=x3, x4=x4 )
    onDia_surf   = ret["surf"]["quad"]
    gmsh.model.occ.synchronize()
    #  -- [6-2] boolean cut of the surface          --  #
    target = [(surfDim,onDia_surf)]
    tools  = [(surfDim,int( tool ) ) for tool in tools ]
    ret    = gmsh.model.occ.cut( target, tools, removeObject=False, removeTool=False )
    gmsh.model.occ.remove( tools, recursive=True )
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()
    #  -- [6-3] rearangement of the volume          --  #
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    # ------------------------------------------------- #
    # --- [7] obtain elemental entity number        --- #
    # ------------------------------------------------- #
    #  -- [7-1] grouping                            --  #
    s_dimtag                 = gmsh.model.getEntities(2)
    surfs                    = [ int( dimtag[1] ) for dimtag in s_dimtag ]
    ceilings, floors, onDias = [], [], []
    for iS,surfnum in enumerate(surfs):
        CoM = gmsh.model.occ.getCenterOfMass( surfDim, surfnum )
        if   ( ( CoM[x_]          > -eps ) and ( CoM[x_]          < +eps ) ):
            onDias  .append( surfnum )
        elif ( ( CoM[z_]          > -eps ) and ( CoM[z_]          < +eps ) ):
            floors  .append( surfnum )
        elif ( ( CoM[z_] - height > -eps ) and ( CoM[z_] - height < +eps ) ):
            ceilings.append( surfnum )
    side_surfs = list( set( surfs ) - set( ceilings ) - set( floors ) - set( onDias ) )
    #  -- [7-2] labeling                             -- #
    z1 = ( gmsh.model.occ.getCenterOfMass( surfDim, side_surfs[0] ) )[z_]
    z2 = ( gmsh.model.occ.getCenterOfMass( surfDim, side_surfs[1] ) )[z_]
    z3 = ( gmsh.model.occ.getCenterOfMass( surfDim, onDias[0] )     )[z_]
    z4 = ( gmsh.model.occ.getCenterOfMass( surfDim, onDias[1] )     )[z_]
    if ( z1 > z2 ):
        side_upper, side_lower = side_surfs[0], side_surfs[1]
    else:
        side_upper, side_lower = side_surfs[1], side_surfs[0]
    if ( z3 > z4 ):
        onDias_upper, onDias_lower = onDias[0], onDias[1]
    else:
        onDias_upper, onDias_lower = onDias[1], onDias[0]
    print( "floors       :: {0}".format( floors       ) )
    print( "ceilings     :: {0}".format( ceilings     ) )
    print( "side_upper   :: {0}".format( side_upper   ) )
    print( "side_lower   :: {0}".format( side_lower   ) )
    print( "onDias_upper :: {0}".format( onDias_upper ) )
    print( "onDias_lower :: {0}".format( onDias_lower ) )

    # ------------------------------------------------- #
    # --- [9] define pole surface                   --- #
    # ------------------------------------------------- #
    arc_pieces_upper = gmsh.model.getBoundary( [(surfDim,side_upper)], oriented=False )
    arc_pieces_lower = gmsh.model.getBoundary( [(surfDim,side_lower)], oriented=False )
    arc_pieces_upper = set( [ int( dimtag[1] ) for dimtag in arc_pieces_upper ] )
    arc_pieces_lower = set( [ int( dimtag[1] ) for dimtag in arc_pieces_lower ] )
    arc_pieces       = list( arc_pieces_upper & arc_pieces_lower )

    ptkeys           = []
    arc_lines        = {}
    arc_points       = []
    for iL,piece in enumerate( arc_pieces ):
        ret     = gmsh.model.getBoundary( [(lineDim,piece)] )
        pt1,pt2 = int( ret[0][1] ), int( ret[1][1] )
        if ( pt1 > pt2 ):
            pt1, pt2 = pt2, pt1
        ptkey            = "{0}_{1}".format( pt1, pt2 )
        arc_lines[ptkey] = piece
        ptkeys.append( ptkey )
        arc_points += [ pt1, pt2 ]
    arc_points = list( set( arc_points ) )

    radii          = np.sqrt( pointData[:,x_]**2 + pointData[:,y_]**2 )
    index          = np.where( ( radius-eps < radii ) & \
                               ( radius+eps > radii ) )
    nodeNum        = ( np.arange( 1, pointData.shape[0]+1 ) )[index]
    onDia_points   = pointData[index]

    # gmsh.model.occ.synchronize()
    # for pt in arc_points:
    #     print( pt )
    #     CoM        = gmsh.model.occ.getCenterOfMass( ptDim, pt )
    #     print( CoM )
    #     distance   = np.sqrt( ( onDia_points[:,x_] - CoM[x_] )**2 + \
    #                           ( onDia_points[:,y_] - CoM[y_] )**2 + \
    #                           ( onDia_points[:,z_] - CoM[z_] )**2 )
    #     closests   = np.argmin( distance )
    #     print( closests )
    # sys.exit()

        
    inpFile        = "dat/mesh.elements"
    with open( inpFile, "r" ) as f:
        connectivities = np.loadtxt( f )
    connectivities = np.array( connectivities[:,3:], dtype=np.int64 )
    connectivities = connectivities - 1

    count = 0
    for ik,cnc in enumerate(connectivities):
        print( ik )
        pt1 = pointData[cnc[0],:]
        pt2 = pointData[cnc[1],:]
        pt3 = pointData[cnc[2],:]
        dr1 = np.sqrt( pt1[x_]**2 + pt1[y_]**2 ) - radius
        dr2 = np.sqrt( pt2[x_]**2 + pt2[y_]**2 ) - radius
        dr3 = np.sqrt( pt3[x_]**2 + pt3[y_]**2 ) - radius
        if   ( ( ( dr2 > -eps ) and ( dr2 < +eps ) ) and ( ( dr3 > -eps ) and ( dr3 < +eps ) ) ):
            ret1 = gmsh.model.occ.addPoint( pt1[x_], pt1[y_], pt1[z_] )
            # ret1 = gmsh.model.getEntitiesInBoundingBox( pt1[x_]-eps, pt1[y_]-eps, pt1[z_]-eps, \
            #                                             pt1[x_]+eps, pt1[y_]+eps, pt1[z_]+eps, \
            #                                             dim=0 )
            ret2 = gmsh.model.getEntitiesInBoundingBox( pt2[x_]-eps, pt2[y_]-eps, pt2[z_]-eps, \
                                                        pt2[x_]+eps, pt2[y_]+eps, pt2[z_]+eps, \
                                                        dim=0 )
            ret3 = gmsh.model.getEntitiesInBoundingBox( pt3[x_]-eps, pt3[y_]-eps, pt3[z_]-eps, \
                                                        pt3[x_]+eps, pt3[y_]+eps, pt3[z_]+eps, \
                                                        dim=0 )
            ret2, ret3 = ret2[0][1], ret3[0][1]
            key        = "{0}_{1}".format( min( ret2, ret3 ), max( ret2, ret3 ) )
            line12     = gmsh.model.occ.addLine( ret1, ret2 )
            line23     = gmsh.model.occ.addLine( ret2, ret3 )
            line31     = gmsh.model.occ.addLine( ret3, ret1 )
            arc23      = arc_lines[key]
            
            l_Group1   = gmsh.model.occ.addCurveLoop( [ +line12, + line23, line31 ] )
            surf1      = gmsh.model.occ.addPlaneSurface( [ l_Group1 ] )
            l_Group2   = gmsh.model.occ.addCurveLoop( [ -line23, + arc23 ] )
            surf2      = gmsh.model.occ.addPlaneSurface( [ l_Group2 ] )
            # print( ret )

        elif ( ( ( dr1 > -eps ) and ( dr1 < +eps ) ) and ( ( dr3 > -eps ) and ( dr3 < +eps ) ) ):
            ret1 = gmsh.model.getEntitiesInBoundingBox( pt1[x_]-eps, pt1[y_]-eps, pt1[z_]-eps, \
                                                        pt1[x_]+eps, pt1[y_]+eps, pt1[z_]+eps, \
                                                        dim=0 )
            ret2 = gmsh.model.occ.addPoint( pt2[x_], pt2[y_], pt2[z_] )
            ret3 = gmsh.model.getEntitiesInBoundingBox( pt3[x_]-eps, pt3[y_]-eps, pt3[z_]-eps, \
                                                        pt3[x_]+eps, pt3[y_]+eps, pt3[z_]+eps, \
                                                        dim=0 )
            # print( ret1, ret3 )
            ret1, ret3 = ret1[0][1], ret3[0][1]
            
        elif ( ( ( dr1 > -eps ) and ( dr1 < +eps ) ) and ( ( dr2 > -eps ) and ( dr2 < +eps ) ) ):
            ret1 = gmsh.model.getEntitiesInBoundingBox( pt1[x_]-eps, pt1[y_]-eps, pt1[z_]-eps, \
                                                        pt1[x_]+eps, pt1[y_]+eps, pt1[z_]+eps, \
                                                        dim=0 )
            ret2 = gmsh.model.getEntitiesInBoundingBox( pt2[x_]-eps, pt2[y_]-eps, pt2[z_]-eps, \
                                                        pt2[x_]+eps, pt2[y_]+eps, pt2[z_]+eps, \
                                                        dim=0 )
            ret3 = gmsh.model.occ.addPoint( pt3[x_], pt3[y_], pt3[z_] )
            ret1, ret2 = ret1[0][1], ret2[0][1]
        
            
    # sys.exit()
    return()




# ========================================================= #
# ===   実行部                                          === #
# ========================================================= #
if ( __name__=="__main__" ):

    # ------------------------------------------------- #
    # --- [1] initialization of the gmsh            --- #
    # ------------------------------------------------- #
    gmsh.initialize()
    gmsh.option.setNumber( "General.Terminal", 1 )
    gmsh.model.add( "model" )
    
    # ------------------------------------------------- #
    # --- [2] define model                          --- #
    # ------------------------------------------------- #
    generate__poleLayer()
    
    # ------------------------------------------------- #
    # --- [4] post process                          --- #
    # ------------------------------------------------- #
    gmsh.option.setNumber( "Mesh.CharacteristicLengthMin", 0.05 )
    gmsh.option.setNumber( "Mesh.CharacteristicLengthMax", 0.05 )
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    # gmsh.write( "msh/model.geo_unrolled" )
    gmsh.write( "msh/model.msh" )
    gmsh.finalize()

