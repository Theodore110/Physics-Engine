
#[derive(Clone,Copy)] struct Body{x:f32,y:f32,vx:f32,vy:f32,inv_m:f32,r:f32}
#[derive(Clone,Copy)] struct Pair{a:u32,b:u32}
#[derive(Clone,Copy,Default)] struct Contact{a:u32,b:u32,nx:f32,ny:f32,bias:f32,jn:f32}
#[no_mangle] pub extern "C" fn k_run(n:i32,steps:i32)->f32{
  let mut st:u32=12345u32;
  let mut rnd=||{st=st.wrapping_mul(1664525).wrapping_add(1013904223);(st>>8) as f32/16777216.0f32};
  let (dt,g,beta,slop)=(1.0f32/60.0,-9.8f32,0.2f32,0.01f32); let iters=10;
  let nu=n as usize;
  let mut bs=vec![Body{x:0.,y:0.,vx:0.,vy:0.,inv_m:0.,r:0.};nu*nu];
  for j in 0..nu{for i in 0..nu{
    let rx=rnd();let ry=rnd();let rv=rnd();
    bs[j*nu+i]=Body{x:i as f32+(rx-0.5)*0.1,y:j as f32+(ry-0.5)*0.1,
      vx:if j==0{0.0}else{(rv-0.5)*0.2},vy:0.0,inv_m:if j==0{0.0}else{1.0},r:0.55};}}
  let mut ps:Vec<Pair>=Vec::new();
  let di=[1i32,0,1,-1];let dj=[0i32,1,1,1];
  for j in 0..n{for i in 0..n{for k in 0..4{
    let ii=i+di[k];let jj=j+dj[k]; if ii<0||ii>=n||jj>=n{continue;}
    if j==0&&jj==0{continue;} ps.push(Pair{a:(j*n+i) as u32,b:(jj*n+ii) as u32});}}}
  let mut cs=vec![Contact::default();ps.len()];
  for _ in 0..steps{
    for b in bs.iter_mut(){ if b.inv_m>0.0{b.vy+=g*dt;} }
    let mut nc=0usize;

    for p in ps.iter(){ unsafe{
      let (a,b)=(*bs.get_unchecked(p.a as usize),*bs.get_unchecked(p.b as usize));
      let dx=b.x-a.x;let dy=b.y-a.y;let d2=dx*dx+dy*dy;let rr=a.r+b.r;
      if d2<rr*rr&&d2>1e-12{
        let d=d2.sqrt();let inv=1.0/d;let mut pen=rr-d-slop; if pen<0.0{pen=0.0;}
        *cs.get_unchecked_mut(nc)=Contact{a:p.a,b:p.b,nx:dx*inv,ny:dy*inv,bias:(beta/dt)*pen,jn:0.0};
        nc+=1; } } }
    for _ in 0..iters{ for c in 0..nc{ unsafe{
      let cp=cs.as_mut_ptr().add(c); let bp=bs.as_mut_ptr();
      let (ai,bi)=((*cp).a as usize,(*cp).b as usize);
      let (ia,ib)=((*bp.add(ai)).inv_m,(*bp.add(bi)).inv_m);
      let vn=((*bp.add(bi)).vx-(*bp.add(ai)).vx)*(*cp).nx+((*bp.add(bi)).vy-(*bp.add(ai)).vy)*(*cp).ny;
      let mut lam=((*cp).bias-vn)/(ia+ib);let mut nj=(*cp).jn+lam;if nj<0.0{nj=0.0;}
      lam=nj-(*cp).jn;(*cp).jn=nj;let px=lam*(*cp).nx;let py=lam*(*cp).ny;
      (*bp.add(ai)).vx-=px*ia;(*bp.add(ai)).vy-=py*ia;(*bp.add(bi)).vx+=px*ib;(*bp.add(bi)).vy+=py*ib; } } }

    for b in bs.iter_mut(){b.x+=b.vx*dt;b.y+=b.vy*dt;} }
  let mut sum=0f64; for b in bs.iter(){sum+=b.x as f64+b.y as f64;} sum as f32
}
