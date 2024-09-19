import numpy as np
import asyncio
import pygame
from pygame.locals import (
    K_r,
    K_SPACE,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)
WIDTH = 500
HEIGHT = 500
n_obstacle = 3
pygame.init()
myfont = pygame.font.SysFont("Arial", 30)


class Node(pygame.sprite.Sprite):

    COLOR_INIT = (34, 227, 147)
    COLOR_CAPTURED = (151, 23, 255)
    COLOR_COLLISION = (255, 0, 0)
    WHITE = (255,255,255)
    DIST = 20
    Kp = 0.1
    Kd = Kp/3.
    BUFFER_SIZE = 20
    R=15

    def __init__(self, x=0, y=0,
                 coef=0.9):
        super().__init__()
        self.surf = pygame.Surface([Node.R*2, Node.R*2])
        self.surf.fill(Node.WHITE)
        self.surf.set_colorkey(Node.WHITE)
        pygame.draw.circle(surface=self.surf,
            color=self.COLOR_INIT,center=(Node.R,Node.R),
            radius=Node.R)
        
        self.rect = self.surf.get_rect()
        self.rect.x = x-Node.R
        self.rect.y = y-Node.R
        self.past_error = 0.0
        self.next = None
        self.buffer = []
        self.target_x = None
        self.target_y = None
        self.history_coef = coef

    @property
    def center(self):
        return self.rect.x+Node.R, self.rect.y+Node.R

    def update(self,x,y, head=True):
        
        if head:
            self._update_head(x,y)
        else:
            self._update(x,y)

    def set_color(self, c):
        pygame.draw.circle(surface=self.surf,
                           color=c,center=(Node.R,Node.R),
                            radius=Node.R)
        
    def _update_head(self, x, y):
        cx,cy = self.center
        if self.target_x is None:
            self.target_x = x
            self.target_y = y
        else:
            self.target_x = self.history_coef*self.target_x + \
                (1-self.history_coef)*x
            self.target_y = self.history_coef*self.target_y + \
                (1-self.history_coef)*y
        x = self.target_x
        y = self.target_y
        
        d = np.sqrt((cx-x)**2 + (cy-y)**2)
        e = np.abs(d-self.DIST)
        de = e-self.past_error
        self.past_error = e
        e_ = (e*Node.Kp + de*Node.Kd)
        dx = (x-cx)/d*e_
        dy = (y-cy)/d*e_
        self.rect.x += dx
        self.rect.y += dy
    
    def _update(self, x, y):
        self.rect.x = x-Node.R
        self.rect.y = y-Node.R

def binary_search(arr, x, l=0):
    """binary search; can supply starting index with l

    Args:
        arr (_type_): _description_
        x (_type_): _description_
        l (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    l = l if l else 0
    r = len(arr)-1
    while l <= r:
        mid = l + (r-l)//2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            l = mid + 1
        else:
            r = mid - 1
    return l

class Snake():
    COLLISION_SELF = 1
    COLLISION_WALL = 2
    COLLISION_OBS = 3
    def __init__(self,init_node,
                 width,height) -> None:
        self.body = pygame.sprite.Group()
        self.head = init_node
        self.head_traj = [list(init_node.center)]
        self.traj_cum_len = [0.0]
        
        self.seg_len = 2.1*Node.R # length of each segment
        self.capture_dist = self.seg_len
        self.n_nodes = 1
        self.canvas_width = width
        self.canvas_height = height

    def update(self,x,y):
        # update head trajectory
        cx,cy = self.head.center
        px,py = self.head_traj[0]
        self.head_traj.insert(0, [cx,cy])

        d = np.sqrt((cx-px)**2 + (cy-py)**2)
        for i, _ in enumerate(self.traj_cum_len):
            self.traj_cum_len[i] += d
        self.traj_cum_len.insert(0, 0.0)

        while self.traj_cum_len[-1] > self.seg_len*(self.n_nodes+1):
            self.traj_cum_len.pop(-1)
            self.head_traj.pop(-1)
            if len(self.head_traj) == 0:
                break

        self.head.update(x,y,head=True)
        next = self.head.next
        ind = 1
        k_prev = None
        while next:
            next:Node
            sl = ind*self.seg_len
            k = binary_search(self.traj_cum_len, sl, k_prev)
            if k<=0 or k>=len(self.traj_cum_len):
                break
            k_prev = k
            # print(f"found {k}")
            # print(len(self.traj_cum_len),len(self.head_traj))
            rho = (sl-self.traj_cum_len[k-1]) / \
                (self.traj_cum_len[k]-self.traj_cum_len[k-1])
            x = (1-rho)*self.head_traj[k-1][0] + \
                rho*self.head_traj[k][0]
            y = (1-rho)*self.head_traj[k-1][1] + \
                rho*self.head_traj[k][1]
            next.update(x,y,head=False)
            next = next.next
            ind+=1

    def capture(self, node : Node):
        """check if head is within some threshold of the rogue
        node

        Args:
            node (_type_): _description_
        """
        cx1,cy1 = self.head.center
        cx2,cy2 = node.center
        d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
        if d < self.capture_dist:
            node.next = self.head
            # move prev head to body
            self.body.add(self.head)
            self.head = node
            node.set_color(Node.COLOR_CAPTURED)
            self.n_nodes += 1
            # print("captured")
            return 1
        return 0
    
    def check_collision(self, all_obs:pygame.sprite.Group):
        """check if head has collided with the body
        """
        if self._check_collision_self():
            return Snake.COLLISION_SELF
        if self._check_collision_wall():
            return Snake.COLLISION_WALL
        if self._check_collision_obs(all_obs):
            return Snake.COLLISION_OBS
        return 0

    def _check_collision_self(self):
        for node in self.body:
            cx1,cy1 = self.head.center
            cx2,cy2 = node.center
            d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 1.0*Node.R:
                self.head.set_color(Node.COLOR_COLLISION)
                node.set_color(Node.COLOR_COLLISION)
                return True
        return False

    def _check_collision_wall(self):
        cx,cy = self.head.center
        if cx < 0 or cx > self.canvas_width or \
            cy < 0 or cy > self.canvas_height:
            self.head.set_color(Node.COLOR_COLLISION)
            return True
        return False

    def _check_collision_obs(self, all_obs):
        for obs in all_obs:
            if pygame.sprite.collide_rect(self.head, obs):
                self.head.set_color(Node.COLOR_COLLISION)
                return True
        return False
    
class NodeSpawner:
    def __init__(self,
                 all_nodes,
                 all_obs,
                 WIDTH, HEIGHT) -> None:
        self.all_nodes = all_nodes
        self.all_obs = all_obs
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

    def spawn(self):
        redo=True
        while redo:
            x=np.random.randint(3*Node.R,self.WIDTH-3*Node.R)
            y=np.random.randint(3*Node.R,self.HEIGHT-3*Node.R)
            node = Node(x,y)
            if pygame.sprite.spritecollideany(node, self.all_nodes) or \
                pygame.sprite.spritecollideany(node, self.all_obs):
                del node
                continue
            else:
                redo=False
        return Node(x,y)
    
class Obstacle(pygame.sprite.Sprite):
    MAX_SIZE = 60
    MIN_SIZE = 20
    COLOR = (252, 3, 223)

    def __init__(self,canvas_width,canvas_height):
        super().__init__()
        self.width = np.random.randint(self.MIN_SIZE, self.MAX_SIZE)
        self.height = np.random.randint(self.MIN_SIZE, self.MAX_SIZE)
        self.surf = pygame.Surface([self.width, self.height])
        self.surf.fill(Obstacle.COLOR)
        self.rect = self.surf.get_rect()
        self.rect.x = np.random.randint(0, canvas_width-self.width)
        self.rect.y = np.random.randint(0, canvas_height-self.height)

    def update(self):
        pass

def wait_for_start(screen, clock):
    screen.fill((255, 255, 255))
    text = myfont.render(f"Press space key to start", 1, (0,0,0))
    screen.blit(text, (0, 0))
    pygame.display.update()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    running = False
            elif event.type == QUIT:
                running = False
        clock.tick(30)

def play(screen, clock):
    running = True

    # spawn obstacles
    all_obs = pygame.sprite.Group()
    all_nodes = pygame.sprite.Group()
    for _ in range(n_obstacle):
        obs = Obstacle(WIDTH, HEIGHT)
        all_obs.add(obs)

    # spawn the snake
    spawner = NodeSpawner(all_nodes,all_obs, WIDTH, HEIGHT)
    init_node = spawner.spawn()
    init_node.set_color(Node.COLOR_CAPTURED)
    all_nodes.add(init_node)
    snake = Snake(init_node, WIDTH, HEIGHT)

    # spawn the first food node
    new_node = spawner.spawn()
    all_nodes.add(new_node)

    # Main loop
    spawn_counter = 1
    capture_counter = 0
    game_over = False
    while running:
        # Look at every event in the queue
        for event in pygame.event.get():
            # Did the user hit a key?
            if event.type == KEYDOWN:
                # Was it the Escape key? If so  , stop the loop.
                if event.key == K_ESCAPE:
                    running = False

            # Did the user click the window close button? If so, stop the loop.
            elif event.type == QUIT:
                running = False

        if snake.capture(new_node):
            capture_counter += 1
        
        if capture_counter >= spawn_counter:
            new_node = spawner.spawn()
            all_nodes.add(new_node)
            spawn_counter += 1

        x, y = pygame.mouse.get_pos()
        snake.update(x,y)

        collision = snake.check_collision(all_obs)
        if collision > 0:
            running = False
            game_over = True
        if collision == Snake.COLLISION_SELF:
            print("collision with self, game over")
        elif collision == Snake.COLLISION_WALL:
            print("collision with wall, game over")
        elif collision == Snake.COLLISION_OBS:
            print("collision with obstacle, game over")
            
        screen.fill((255, 255, 255))
        text = myfont.render(f"Score: {snake.n_nodes}", 1, (0,0,0))
        screen.blit(text, (0, 0))
        
        for obs in all_obs:
            screen.blit(obs.surf, obs.rect)

        for node in all_nodes:
            screen.blit(node.surf, node.rect)

        if game_over:
            text = myfont.render(f"Game Over", 1, (255,0,0))
            screen.blit(text, (WIDTH//2-50, HEIGHT//2-20))
        pygame.display.update()
        clock.tick(30)

def exit_game(screen, clock):
    screen.fill((255, 255, 255))
    text = myfont.render(f"Press R (reset) / Esc (exit)", 1, (0,0,0))
    screen.blit(text, (0, 0))
    pygame.display.update()
    running = True
    exit=False
    while running:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                    exit=True
                elif event.key == K_r:
                    running = False
                    exit=False
            elif event.type == QUIT:
                running = False
                exit=True
        clock.tick(30)
    return exit

screen = pygame.display.set_mode([WIDTH, HEIGHT])
clock = pygame.time.Clock()
wait_for_start(screen, clock)
while True:
    play(screen, clock)
    pygame.time.wait(2000)
    if exit_game(screen, clock):
        break

pygame.quit()