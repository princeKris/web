import { Component, OnInit } from '@angular/core';
import { SharedImports } from '../shared/shared-imports';
import { ActivatedRoute, Router } from '@angular/router';
import { ProfileService } from '../services/profile-service';
import { formatDistanceToNow } from 'date-fns';
import { Auth } from '../services/auth';
@Component({
  selector: 'app-viwe-profile',
  imports: [...SharedImports],
  templateUrl: './viwe-profile.html',
  styleUrl: './viwe-profile.scss',
})
export class ViweProfile implements OnInit {
  user: any = null;
  posts: any[] = [];
  previewImage: string | null = null;
  totalLikes=0;
  totalComment=0;

constructor(private route: ActivatedRoute,private pro:ProfileService, private auth:Auth,private router:Router) {}

ngOnInit() {
  const id = this.route.snapshot.paramMap.get('id');
  console.log("PROFILE ID:", id);

  if (id) {
    this.pro.getProfileById(Number(id)).subscribe({
      next: (value:any) =>{
        this.user = value.user;     // <-- correct
        this.posts = value.posts; 
        this.totalLikes = this.posts.reduce((sum:any, p:any) => sum + p.likes_count, 0);
        this.totalComment = this.posts.reduce((sum:any, p:any) => sum + p.comments.length, 0);
        console.log(this.user);
      },
      error(err) {
        console.log(err);
      },
    });
  }
}


onToggleLike(posts:any,id:number) {
    this.pro.likePost(id).subscribe({
      next(value:any) {
        posts.is_liked=value.liked;
        posts.likes_count=value.likes_count;
        console.log(value);
      },
      error(err) {
        const backendMessage =
    err.error?.error ||
    err.error?.detail ||
    err.error?.message ||
    "Something went wrong";
    if (err.status === 401) { 
      alert("Please sign in to like a post."); 
    } else if (err.status === 403) { 
      alert("You do not have permission to like."); 
    } else if (err.status === 400) { 
      alert("Invalid like."); 
    } else { 
      alert(backendMessage); 
    }
      },
    });
  }


getRelativeDate(date: string) {
    return formatDistanceToNow(new Date(date), { addSuffix: true });
  }


  deleteMyAccount(id:number) {
    console.log(id);
  if (!confirm("Are you sure you want to delete The account? This cannot be undone.")) {
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
      
    }
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

toggleVerify(user: any) {
  const action = user.verified ? "unverify" : "verify";

  if (!confirm(`Are you sure you want to ${action} this user?`)) return;

  this.pro.toggleVerify(user.id).subscribe({
    next: (res: any) => {
      alert(res.message);
      user.verified = !user.verified; // UPDATE UI INSTANTLY
    },
    error: (err) => {
      console.error(err);
      alert(err.error?.error || "Something went wrong");
    }
  });
}

}
