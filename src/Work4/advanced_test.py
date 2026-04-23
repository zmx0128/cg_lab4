import taichi as ti # type: ignore

# 初始化 Taichi
ti.init(arch=ti.gpu)

# 窗口分辨率
res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 定义全局交互参数
Ka = ti.field(ti.f32, shape=())
Kd = ti.field(ti.f32, shape=())
Ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())
use_blinn_phong = ti.field(ti.i32, shape=())
use_shadow = ti.field(ti.i32, shape=())

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

# --- 几何体相交测试函数 ---

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """测试光线与球体相交"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    """
    测试光线与竖直圆锥相交
    apex: 圆锥顶点坐标
    base_y: 圆锥底面的世界坐标 Y 值
    """
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    H = apex.y - base_y
    k = (radius / H) ** 2
    
    # 转换到以顶点为原点的局部坐标系
    ro_local = ro - apex
    
    # 构建一元二次方程 At^2 + Bt + C = 0
    A = rd.x**2 + rd.z**2 - k * rd.y**2
    B = 2.0 * (ro_local.x * rd.x + ro_local.z * rd.z - k * ro_local.y * rd.y)
    C = ro_local.x**2 + ro_local.z**2 - k * ro_local.y**2
    
    # 避免 A 为 0 时的除零错误
    if ti.abs(A) > 1e-5:
        delta = B**2 - 4.0 * A * C
        if delta > 0:
            t1 = (-B - ti.sqrt(delta)) / (2.0 * A)
            t2 = (-B + ti.sqrt(delta)) / (2.0 * A)
            
            # 保证 t_first 是较近的交点
            t_first = t1
            t_second = t2
            if t1 > t2:
                t_first, t_second = t_second, t_first
                
            # 验证交点是否在圆锥的高范围内 (局部 Y 坐标在 [-H, 0] 之间)
            y1 = ro_local.y + t_first * rd.y
            if t_first > 0 and -H <= y1 <= 0:
                t = t_first
            else:
                y2 = ro_local.y + t_second * rd.y
                if t_second > 0 and -H <= y2 <= 0:
                    t = t_second
                    
            if t > 0:
                p_local = ro_local + rd * t
                # 圆锥表面的法线计算
                normal = normalize(ti.Vector([p_local.x, -k * p_local.y, p_local.z]))
                
    return t, normal

@ti.func
def intersect_scene(ro, rd):
    """测试光线与场景中所有物体的相交"""
    min_t = 1e10
    hit_normal = ti.Vector([0.0, 0.0, 0.0])
    hit_color = ti.Vector([0.0, 0.0, 0.0])
    
    # 1. 测试与红球相交
    t_sph, n_sph = intersect_sphere(ro, rd, ti.Vector([-0.5, -0.2, 0.0]), 1.0)
    if 0 < t_sph < min_t:
        min_t = t_sph
        hit_normal = n_sph
        hit_color = ti.Vector([0.8, 0.1, 0.1])
        
    # 2. 测试与紫色圆锥相交
    t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([1.0, 1.0, 0.0]), -1.2, 1.0)
    if 0 < t_cone < min_t:
        min_t = t_cone
        hit_normal = n_cone
        hit_color = ti.Vector([0.6, 0.2, 0.8])
    
    return min_t, hit_normal, hit_color

@ti.func
def is_in_shadow(p, light_pos):
    """判断点 p 是否在阴影中"""
    shadow_ray_dir = normalize(light_pos - p)
    shadow_ray_origin = p + shadow_ray_dir * 1e-3  # 避免自遮挡
    t_light = (light_pos - p).norm()
    
    t, _, _ = intersect_scene(shadow_ray_origin, shadow_ray_dir)
    return t > 0 and t < t_light

@ti.kernel
def render():
    for i, j in pixels:
        u = (i - res_x / 2.0) / res_y * 2.0
        v = (j - res_y / 2.0) / res_y * 2.0
        
        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))

        # 用于记录光线击中的最近物体
        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])
        
        # 测试与场景中所有物体的相交
        min_t, hit_normal, hit_color = intersect_scene(ro, rd)

        # 背景色
        color = ti.Vector([0.25, 0.25, 0.25]) 

        # 如果击中了任何物体
        if min_t < 1e9:
            p = ro + rd * min_t
            N = hit_normal
            
            # 光源设置
            light_pos = ti.Vector([-3.0, 2.0, 2.0])
            light_color = ti.Vector([1.0, 1.0, 1.0]) 
            
            L = normalize(light_pos - p)
            V = normalize(ro - p)

            # 检查是否在阴影中
            in_shadow = False
            if use_shadow[None]:
                in_shadow = is_in_shadow(p, light_pos)
            
            if in_shadow:
                # 在阴影中，只计算环境光
                ambient = Ka[None] * light_color * hit_color
                color = ambient
            else:
                # --- 光照模型计算 ---
                ambient = Ka[None] * light_color * hit_color
                
                diff = ti.max(0.0, N.dot(L))
                diffuse = Kd[None] * diff * light_color * hit_color
                
                spec = 0.0
                if use_blinn_phong[None]:
                    # Blinn-Phong 模型
                    H = normalize(L + V)
                    spec = ti.max(0.0, N.dot(H)) ** shininess[None]
                else:
                    # Phong 模型
                    R = normalize(reflect(-L, N))
                    spec = ti.max(0.0, R.dot(V)) ** shininess[None]
                
                specular = Ks[None] * spec * light_color 
                
                color = ambient + diffuse + specular
                    
        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Advanced Phong Shading Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    # 初始化材质参数
    ka = 0.2
    kd = 0.7
    ks = 0.5
    shininess_val = 32.0
    blinn_phong = True
    shadow = True
    
    # 初始更新 Taichi 字段
    Ka[None] = ka
    Kd[None] = kd
    Ks[None] = ks
    shininess[None] = shininess_val
    use_blinn_phong[None] = 1 if blinn_phong else 0
    use_shadow[None] = 1 if shadow else 0

    while window.running:
        # 绘制交互面板
        with gui.sub_window("Material Parameters", 0.7, 0.05, 0.28, 0.35):
            ka = gui.slider_float('Ka (Ambient)', ka, 0.0, 1.0)
            kd = gui.slider_float('Kd (Diffuse)', kd, 0.0, 1.0)
            ks = gui.slider_float('Ks (Specular)', ks, 0.0, 1.0)
            shininess_val = gui.slider_float('N (Shininess)', shininess_val, 1.0, 128.0)
            blinn_phong = gui.checkbox('Blinn-Phong Model', blinn_phong)
            shadow = gui.checkbox('Hard Shadow', shadow)
        
        # 更新 Taichi 字段
        Ka[None] = ka
        Kd[None] = kd
        Ks[None] = ks
        shininess[None] = shininess_val
        use_blinn_phong[None] = 1 if blinn_phong else 0
        use_shadow[None] = 1 if shadow else 0
        
        # 执行并行渲染
        render()
        
        # 将渲染结果绘制到画布
        canvas.set_image(pixels)

        # 显示窗口
        window.show()

if __name__ == '__main__':
    main()