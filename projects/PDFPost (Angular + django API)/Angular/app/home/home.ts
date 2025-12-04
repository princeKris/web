import { Component, OnInit, OnDestroy } from '@angular/core';
import { SharedImports } from '../shared/shared-imports';
import { Router } from '@angular/router';
import { formatDistanceToNow } from 'date-fns';
import { ProfileService } from '../services/profile-service';
import { Auth } from '../services/auth';

/* ------------------------------
   INTERFACES
------------------------------ */
interface Author {
  id: number;
  username: string;
  profilePic: string | null;
}

interface Post {
  id: number;
  title: string;
  description: string;
  image: string | null;
  file: string;
  updated_at: string;
  likes_count: number;
  is_liked: boolean;
  author: Author;
}

@Component({
  selector: 'app-home',
  imports: [...SharedImports],
  templateUrl: './home.html',
  styleUrl: './home.scss',
})
export class Home implements OnInit, OnDestroy {
  
  /* ------------------------------ */
  /* DATA */
  /* ------------------------------ */
  post:any;

posts: any[] = [];
page = 1;
limit = 10;
hasMore = true;
loading = false;
observer?: IntersectionObserver;



topusers:any;
topposts:any;
  constructor(private router: Router, private pro:ProfileService, private auth: Auth) {}

  /* ------------------------------
     LIFECYCLE
  ------------------------------ */

  ngOnInit(): void {
    // this.pro.getAllHomePost().subscribe({
    //   next: (data) => {
    //     this.post=data;
    //     console.log(this.post);
    //   },
    //   error: (err) => {
    //     console.log(err);
    //   }
    // });
    this.pro.getTopUsersByLikes().subscribe({
      next:(value:any)=> {
        this.topusers=value;
      },
    });
     this.pro.getTopPosts().subscribe({
      next: (data) => {
        this.topposts = data;
        console.log("TOP POSTS:", data);
      },
      error: (err) => {
        console.error(err);
      }
    });
    this.loadPosts();     // Load first 10 posts
    this.setupInfiniteScroll();
  }

  ngOnDestroy(): void {
    this.observer?.disconnect();
  }

  /* ------------------------------
     INFINITE SCROLL
  ------------------------------ */

  setupInfiniteScroll() {
  this.observer = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting && this.hasMore) {
      this.loadPosts();
    }
  }, { rootMargin: "200px" });

  setTimeout(() => {
    const sentinel = document.getElementById("scroll-sentinel");
    if (sentinel) this.observer?.observe(sentinel);
  }, 500);
}

 loadPosts() {
  if (this.loading || !this.hasMore) return;

  this.loading = true;

  this.pro.getHomePosts(this.page, this.limit).subscribe({
    next: (res) => {
      this.posts = [...this.posts, ...res.results];
      this.hasMore = res.has_more;
      this.page++; // next page
      this.loading = false;
    },
    error: () => { this.loading = false; }
  });
}

  /* ------------------------------
     HELPERS
  ------------------------------ */

  goToDetails(item: { id: number }) {
  this.router.navigate(['/post', item.id]);
}

  readMore(post: Post) {
    this.goToDetails(post);
  }

  onToggleLike(post: Post,id:number) {
    this.pro.likePost(id).subscribe({
      next(value:any) {
        post.is_liked=value.liked;
        post.likes_count=value.likes_count;
        console.log(value);
      },
      error(err) {
        console.log(err);
      },
    });
  }


  getRelativeDate(date: string) {
    return formatDistanceToNow(new Date(date), { addSuffix: true });
  }


  deleteMyAccount(id:number) {
  if (!confirm("Are you sure you want to delete your account? This cannot be undone.")) {
    return;
  }

  this.pro.deleteAccount(id).subscribe({
    next: () => {
      const u=this.auth.getUser();
      if(u.role==="admin" || u.role==="staff" ){
        alert("account has been deleted");
      }else{
        localStorage.clear();
        this.router.navigate(['/home']);
      }
      
    },
     error:(err) =>{
      const backendMessage =
    err.error?.error ||
    err.error?.detail ||
    err.error?.message ||
    "Something went wrong";
    if (err.status === 401) { 
      alert("Please sign in to delete a post."); 
    } else if (err.status === 403) { 
      alert("You do not have permission to delete the post."); 
    } else if (err.status === 400) { 
      alert("Invalid option."); 
    } else { 
      alert(backendMessage); 
    }
    },
  });
}


deletePost(id:number){
  if (!confirm("Are you sure you want to delete post? This cannot be undone.")) {
    return;
  }
  this.pro.deleteMyPost(id).subscribe({
    next:(data)=>{
      alert("post sucessfully deleted");
      this.posts = this.posts.filter((p: any) => p.id !== id);
    },
    error:(err) =>{
      const backendMessage =
    err.error?.error ||
    err.error?.detail ||
    err.error?.message ||
    "Something went wrong";
    if (err.status === 401) { 
      alert("Please sign in to delete a post."); 
    } else if (err.status === 403) { 
      alert("You do not have permission to delete the post."); 
    } else if (err.status === 400) { 
      alert("Invalid option."); 
    } else { 
      alert(backendMessage); 
    }
    },
  });
}
avaliableFor(id:number){
  const u=this.auth.getUser();
  if(u){
    if(u.role==="admin" || u.role==="staff" || id === u.id){
    return true;
    }else{
      return false;
    }
  }else{
    return false;
  }
  
}

avaliableForA(id:number){
  const u=this.auth.getUser();
  if(u){
    if(u.role==="admin"){
    return true;
    }else{
      return false;
    }
  }else{
    return false;
  }
  
}

}
