"use client";

import { User } from "lucide-react";
import Link from "next/link";
import { useSession } from "next-auth/react";
import { Button } from "./button";

export function SignInButton() {
  const { data: session, status } = useSession();

  return (
    <Button variant="ghost" size="sm" asChild>
      <Link href="/auth/signin" className="flex items-center">
        <User className="mr-2 h-4 w-4" />
        <span>
          {status === "loading"
            ? "Loading..."
            : session
              ? session.user?.email || "Signed In"
              : "Sign In"}
        </span>
      </Link>
    </Button>
  );
}
