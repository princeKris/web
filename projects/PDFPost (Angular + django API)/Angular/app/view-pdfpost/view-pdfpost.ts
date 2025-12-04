import { Component, OnInit } from '@angular/core';
import { SharedImports } from '../shared/shared-imports';
import { ActivatedRoute } from '@angular/router';
import { ProfileService } from '../services/profile-service';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { Auth } from '../services/auth';
@Component({
  selector: 'app-view-pdfpost',
  imports: [...SharedImports],
  templateUrl: './view-pdfpost.html',
  styleUrl: './view-pdfpost.scss',
})
export class ViewPdfpost implements OnInit {
 post: any = null;
  newComment = "";
  safePdfUrl!: SafeResourceUrl;
   constructor(private route: ActivatedRoute, private api: ProfileService,private sanitizer: DomSanitizer,private auth:Auth) {}

  ngOnInit() {
    const id = this.route.snapshot.paramMap.get("id");

    if (id) {
      this.api.getSinglePost(Number(id)).subscribe({
        next: (data:any) => {
          this.post = data;
          this.safePdfUrl = this.sanitizer.bypassSecurityTrustResourceUrl(this.post.file);
          console.log(this.post);
        }
      });
    }
  }

  addComment() {
    if (!this.newComment.trim()) return;

    this.api.addComment(this.post.id, { text: this.newComment }).subscribe({
      next: (c:any) => {
        this.post.comments.unshift(c);
        this.newComment = "";
      },
      error:(err)=> {
        if (err.status === 401) {
          alert("Please sign in to post a comment.");
        } else if (err.status === 403) {
          alert("You do not have permission to comment.");
        } else if (err.status === 400) {
          alert("Invalid comment.");
        } else {
          alert("Something went wrong!");
        }
      },
    });
  }



  onToggleLike(posts:any,id:number) {
    this.api.likePost(id).subscribe({
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
      alert("Please sign n to like a post."); 
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

canDeleteComment(c: any): boolean {
  const currentUser = this.auth.getUser();
  if(currentUser){
    return (
    c.user?.id === currentUser.id ||     // comment owner
    this.post?.author?.id === currentUser.id ||// post owner
    currentUser.role === "admin" ||
    currentUser.role === "staff"
  );
  }else{
    return false;
  }
  
}


deleteComment(id:number) {
  this.api.deleteComment(this.post.id, id).subscribe({
    next: () => {
      alert("comment deleted successfuly");
      this.post.comments = this.post.comments.filter((x: any) => x.id !== id);
    },
    error(err) {
        const backendMessage =
    err.error?.error ||
    err.error?.detail ||
    err.error?.message ||
    "Something went wrong";
    if (err.status === 401) { 
      alert("Please sign in to delete a comment."); 
    } else if (err.status === 403) { 
      alert("You do not have permission to delete comment."); 
    } else if (err.status === 400) { 
      alert("Invalid comment."); 
    } else { 
      alert(backendMessage); 
    }
      },
  });
  //this.post.comments.unshift(c);
  this.newComment = "";
}


}
